// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com

//! Whether a stretch of audio holds a voice, speaking or singing, by a small
//! sound-type classifier (CED tiny, trained on AudioSet) run on the CPU.
//!
//! The speech detector (Silero) is trained on talk and scores singing as
//! nothing, so sung lines never reached transcription. This tagger labels
//! speech and singing directly and tells them from instrumental music. On
//! stored session audio (2026-10-12), with "at least 3 of the 2 s windows of a
//! cut score 0.04 or more", sung cuts passed 20 of 24 on system audio and 44
//! of 45 on a microphone (the detector: 9 and 13), speech 42 of 43, and no
//! instrumental cut; about 40 ms of CPU per 8 s cut.
//!
//! The model takes 64 mel bands per 10 ms, computed as kaldi-native-fbank does
//! with CED's options (sherpa-onnx `offline-stream.cc`, CEDTag): 32 ms periodic
//! Hann frames, no dither, pre-emphasis or DC removal, frames centred on the
//! signal (`snip_edges=false`, reflected at the ends), power spectrum, mel
//! bands 0–8000 Hz, then decibels clamped to 120 dB under the loudest value.

use anyhow::{Context, Result};
use realfft::RealFftPlanner;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{info, warn};

const SAMPLE_RATE: usize = 16_000;
const FRAME: usize = 512; // 32 ms
const SHIFT: usize = 160; // 10 ms
const MELS: usize = 64;
/// The tagger looks at 2 s at a time, one window per second.
pub const WINDOW_SECS: usize = 2;
/// A window this likely to hold speech or singing counts as voiced.
pub const VOICE_SCORE: f32 = 0.04;
/// Voiced windows a cut needs (fewer when the cut has fewer windows): one
/// window over the line is what an instrumental spike looks like.
pub const VOICED_WINDOWS: usize = 3;

/// AudioSet classes: speech of any kind, whispering.
const SPEECH: [usize; 7] = [0, 1, 2, 3, 4, 5, 15];
/// Singing, choir, yodeling, chant, male/female/child/synthetic singing,
/// rapping, humming. Summed: a sung line spreads over several of them.
const SINGING: [usize; 10] = [27, 28, 29, 30, 32, 33, 34, 35, 36, 37];

pub struct SoundTagger {
    session: Mutex<ort::session::Session>,
    mel: Vec<Vec<(usize, f64)>>,
}

/// The installed model: `sound-tags/ced-tiny.int8.onnx` beside the recorder's
/// binary, or `SCREENPIPE_SOUND_TAGS_MODEL`. Loaded once; `None` when it is
/// missing or does not load, and the callers then behave as before.
pub fn tagger() -> Option<Arc<SoundTagger>> {
    static TAGGER: OnceLock<Option<Arc<SoundTagger>>> = OnceLock::new();
    TAGGER
        .get_or_init(|| {
            let path = std::env::var_os("SCREENPIPE_SOUND_TAGS_MODEL")
                .map(PathBuf::from)
                .or_else(|| {
                    let exe = std::env::current_exe().ok()?;
                    Some(exe.parent()?.join("sound-tags").join("ced-tiny.int8.onnx"))
                })?;
            if !path.is_file() {
                info!(
                    "sound tags: no model at {}; singing is not detected",
                    path.display()
                );
                return None;
            }
            match SoundTagger::load(&path) {
                Ok(tagger) => Some(Arc::new(tagger)),
                Err(e) => {
                    warn!("sound tags: model did not load ({e:#}); singing is not detected");
                    None
                }
            }
        })
        .clone()
}

impl SoundTagger {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let session = crate::speaker::create_session(path).context("ced session")?;
        Ok(Self {
            session: Mutex::new(session),
            mel: mel_banks(),
        })
    }

    /// One score per second of `samples_16k` (2 s windows, 1 s apart; one
    /// window for shorter audio): how likely the window holds speech, or
    /// singing (its classes summed), whichever is higher.
    pub fn voice_scores(&self, samples_16k: &[f32]) -> Result<Vec<f32>> {
        let window = WINDOW_SECS * SAMPLE_RATE;
        let mut scores = Vec::new();
        let mut start = 0;
        loop {
            let end = (start + window).min(samples_16k.len());
            if end - start < FRAME {
                break;
            }
            scores.push(self.score(&samples_16k[start..end])?);
            if end == samples_16k.len() {
                break;
            }
            start += SAMPLE_RATE;
        }
        Ok(scores)
    }

    fn score(&self, samples: &[f32]) -> Result<f32> {
        let feats = log_mel(samples, &self.mel);
        let frames = feats.len() / MELS;
        // [1, MELS, frames]: the model wants bands first.
        let mut input = ndarray::Array3::<f32>::zeros((1, MELS, frames));
        for (t, row) in feats.chunks(MELS).enumerate() {
            for (m, v) in row.iter().enumerate() {
                input[[0, m, t]] = *v;
            }
        }
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let inputs = ort::inputs!["feats" => ort::value::TensorRef::from_array_view(&input)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?];
        let outputs = session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("ort: {e}"))?;
        let probs = outputs
            .get("prob")
            .context("no prob output")?
            .try_extract_array::<f32>()
            .context("prob tensor")?;
        let p: Vec<f32> = probs.iter().copied().collect();
        let speech = SPEECH
            .iter()
            .map(|&i| p.get(i).copied().unwrap_or(0.0))
            .fold(0.0, f32::max);
        let singing: f32 = SINGING
            .iter()
            .map(|&i| p.get(i).copied().unwrap_or(0.0))
            .sum();
        Ok(speech.max(singing))
    }
}

/// `true` when a cut's scores show a voice: at least [`VOICED_WINDOWS`]
/// windows at [`VOICE_SCORE`] or more, or all of them for a shorter cut.
pub fn has_voice(scores: &[f32]) -> bool {
    let voiced = scores.iter().filter(|&&s| s >= VOICE_SCORE).count();
    !scores.is_empty() && voiced >= VOICED_WINDOWS.min(scores.len())
}

/// The voiced stretches of a longer recording, in seconds: windows at
/// [`VOICE_SCORE`] or more that have at least [`VOICED_WINDOWS`] such windows
/// within the 8 s around them (the live gate's cut), merged.
pub fn voiced_stretches(scores: &[f32]) -> Vec<(f64, f64)> {
    let voiced: Vec<bool> = scores.iter().map(|&s| s >= VOICE_SCORE).collect();
    let mut stretches: Vec<(f64, f64)> = Vec::new();
    for i in 0..voiced.len() {
        let near = voiced[i.saturating_sub(3)..(i + 4).min(voiced.len())]
            .iter()
            .filter(|&&v| v)
            .count();
        if !voiced[i] || near < VOICED_WINDOWS {
            continue;
        }
        let (start, end) = (i as f64, (i + WINDOW_SECS) as f64);
        match stretches.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => stretches.push((start, end)),
        }
    }
    stretches
}

/// The mel filter bank, kaldi's way: triangles evenly spaced on
/// mel = 1127 ln(1 + f / 700), over the FFT bins below Nyquist.
fn mel_banks() -> Vec<Vec<(usize, f64)>> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let (low, high) = (mel(0.0), mel(SAMPLE_RATE as f64 / 2.0));
    let step = (high - low) / (MELS + 1) as f64;
    let bin_hz = SAMPLE_RATE as f64 / FRAME as f64;
    (0..MELS)
        .map(|b| {
            let (left, centre, right) = (
                low + b as f64 * step,
                low + (b + 1) as f64 * step,
                low + (b + 2) as f64 * step,
            );
            (0..FRAME / 2)
                .filter_map(|i| {
                    let m = mel(bin_hz * i as f64);
                    if m <= left || m >= right {
                        return None;
                    }
                    let w = if m <= centre {
                        (m - left) / (centre - left)
                    } else {
                        (right - m) / (right - centre)
                    };
                    Some((i, w))
                })
                .collect()
        })
        .collect()
}

/// Mel band energies in decibels, `frames × MELS`, row-major.
fn log_mel(samples: &[f32], mel: &[Vec<(usize, f64)>]) -> Vec<f32> {
    let n = samples.len() as i64;
    let frames = (samples.len() + SHIFT / 2) / SHIFT;
    let window: Vec<f64> = (0..FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
        .collect();
    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(FRAME);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut out = Vec::with_capacity(frames * MELS);
    for f in 0..frames {
        let first = (f * SHIFT + SHIFT / 2) as i64 - (FRAME / 2) as i64;
        for (k, slot) in input.iter_mut().enumerate() {
            let mut s = first + k as i64;
            while s < 0 || s >= n {
                s = if s < 0 { -s - 1 } else { 2 * n - 1 - s };
            }
            *slot = samples[s as usize] as f64 * window[k];
        }
        // A fixed-size plan on buffers of its own size cannot fail.
        let _ = fft.process(&mut input, &mut spectrum);
        for bank in mel {
            let energy: f64 = bank.iter().map(|&(i, w)| w * spectrum[i].norm_sqr()).sum();
            out.push(energy as f32);
        }
    }
    let mut top = f32::MIN;
    for v in out.iter_mut() {
        *v = 10.0 * v.max(1e-10).log10();
        top = top.max(*v);
    }
    for v in out.iter_mut() {
        *v = v.max(top - 120.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A signal anyone can rebuild: two tones and a slow chirp.
    fn test_signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                let t = n as f32 / SAMPLE_RATE as f32;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * (200.0 + 1500.0 * t) * t).sin()
                    + 0.05 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn features_match_kaldi_native_fbank() {
        // Reference: kaldi_native_fbank (Python) with CED's options on the
        // same signal, before decibels; frames 0, 50 and 99, bands 0, 10, 40, 63.
        let feats = log_mel(&test_signal(16_000), &mel_banks());
        assert_eq!(feats.len() / MELS, 100);
        for &(frame, band, want_db) in REFERENCE_DB {
            let got = feats[frame * MELS + band];
            assert!(
                (got - want_db).abs() < 0.01,
                "frame {frame} band {band}: {got} vs {want_db}"
            );
        }
    }

    // kaldi_native_fbank 1.x (Python), FbankOptions as in the module comment,
    // on `test_signal(16_000)` in float32, then the same decibel step.
    const REFERENCE_DB: &[(usize, usize, f32)] = &[
        (0, 0, 13.0500),
        (0, 10, 23.3117),
        (0, 40, -4.9149),
        (0, 63, -39.1137),
        (50, 0, -53.6815),
        (50, 10, 17.4913),
        (50, 40, -67.7396),
        (50, 63, -59.5179),
        (99, 0, 8.6125),
        (99, 10, 24.1656),
        (99, 40, 2.8122),
        (99, 63, -32.9394),
    ];

    #[test]
    fn a_cut_needs_three_voiced_windows_or_all_of_a_short_one() {
        assert!(has_voice(&[0.0, 0.05, 0.0, 0.2, 0.0, 0.04, 0.0]));
        assert!(!has_voice(&[0.0, 0.05, 0.0, 0.2, 0.0, 0.03, 0.0]));
        assert!(!has_voice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11])); // one spike, as in an instrumental
        assert!(has_voice(&[0.06]));
        assert!(has_voice(&[0.06, 0.05]));
        assert!(!has_voice(&[0.06, 0.01]));
        assert!(!has_voice(&[]));
    }

    #[test]
    fn stretches_merge_and_drop_lone_windows() {
        let mut scores = vec![0.0; 30];
        for s in &mut scores[5..12] {
            *s = 0.1;
        }
        scores[25] = 0.5; // alone
        assert_eq!(voiced_stretches(&scores), vec![(5.0, 13.0)]);
    }

    #[test]
    fn a_window_per_second() {
        let Ok(path) = std::env::var("SCREENPIPE_SOUND_TAGS_MODEL") else {
            return; // the model is not part of the repository
        };
        let tagger = SoundTagger::load(std::path::Path::new(&path)).unwrap();
        assert_eq!(
            tagger.voice_scores(&test_signal(8 * 16_000)).unwrap().len(),
            7
        );
        assert_eq!(tagger.voice_scores(&test_signal(16_000)).unwrap().len(), 1);
        assert!(tagger.voice_scores(&[]).unwrap().is_empty());
    }
}
