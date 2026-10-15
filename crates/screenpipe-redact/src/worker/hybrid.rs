// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com

use super::Worker;
use crate::{RedactError, RedactionOutput, Redactor};
use screenpipe_db::storage::{FramePayload, PrivacyPolicy, Projection, StorageMode};
use sqlx::Row;
use std::sync::Arc;

/// Retries of one frame read invalidated by a concurrent write.
const READ_ATTEMPTS: u64 = 8;

fn is_revision_change(error: &sqlx::Error) -> bool {
    matches!(error, sqlx::Error::Protocol(message) if message.starts_with("frame storage: read revision changed;"))
}

struct CompleteRedactor<'a>(&'a dyn Redactor);

#[async_trait::async_trait]
impl Redactor for CompleteRedactor<'_> {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn version(&self) -> u32 {
        self.0.version()
    }
    async fn redact_batch(&self, texts: &[String]) -> Result<Vec<RedactionOutput>, RedactError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.0.redact_complete(text).await?);
        }
        Ok(results)
    }
}

/// Frames still owing a configured surface. Under the drain gate: staged
/// frames first (they block sealing and hold the staging budget), then oldest
/// first, so the backlog of already-sealed frames a policy change requeues
/// wholesale waits behind the frames that are actually stuck. Ungated:
/// upstream's newest-first.
const PENDING_FRAMES_NEWEST_FIRST: &str = "SELECT p.frame_id FROM frame_payloads p CROSS JOIN storage_metadata m WHERE (p.completed_surfaces & m.required_surfaces)!=m.required_surfaces AND (p.retry_at IS NULL OR p.retry_at<=strftime('%s','now')) ORDER BY p.frame_id DESC LIMIT ?";
const PENDING_FRAMES_STAGED_OLDEST_FIRST: &str = "SELECT p.frame_id FROM frame_payloads p CROSS JOIN storage_metadata m WHERE (p.completed_surfaces & m.required_surfaces)!=m.required_surfaces AND (p.retry_at IS NULL OR p.retry_at<=strftime('%s','now')) ORDER BY (p.state='staged') DESC, p.frame_id ASC LIMIT ?";

impl Worker {
    pub fn with_frame_storage(mut self, db: Arc<screenpipe_db::DatabaseManager>) -> Self {
        if db.storage_mode() == StorageMode::HybridParquetV1 {
            self.frame_storage = Some(db);
        }
        self
    }

    pub async fn process_hybrid_frames(&self, limit: u32) -> Result<u32, anyhow::Error> {
        let Some(db) = &self.frame_storage else {
            return Ok(0);
        };
        let columns = self.cfg.columns;
        let required = 1
            | 8
            | if columns.accessibility_text { 2 } else { 0 }
            | if columns.accessibility_tree { 4 } else { 0 }
            | if columns.window_name { 16 } else { 0 }
            | if columns.browser_url { 32 } else { 0 };
        let identity = format!("{}:{columns:?}", self.redactor.policy_identity());
        db.set_frame_privacy_policy(&PrivacyPolicy {
            identity: identity.clone(),
            required_surfaces: required,
        })
        .await?;
        let query = if super::drain::is_gated() {
            PENDING_FRAMES_STAGED_OLDEST_FIRST
        } else {
            PENDING_FRAMES_NEWEST_FIRST
        };
        let ids: Vec<i64> = sqlx::query_scalar(query)
            .bind(limit)
            .fetch_all(&db.pool)
            .await?;
        let parallelism = super::drain::parallelism();
        let mut count = 0;
        let mut first_error = None;
        if parallelism <= 1 || ids.len() <= 1 {
            for id in ids {
                if self
                    .process_hybrid_frame(db, id, &identity, required)
                    .await?
                {
                    count += 1;
                }
            }
        } else {
            // Draining: keep `parallelism` frames in flight. Each task blocks
            // its runtime thread inside the ONNX call (`block_in_place`), so
            // this only pays off with as many sessions in the adapter's pool.
            // After a failure no new frame starts; running ones finish.
            let mut ids = ids.into_iter();
            let mut tasks = tokio::task::JoinSet::new();
            loop {
                while first_error.is_none() && tasks.len() < parallelism {
                    let Some(id) = ids.next() else { break };
                    let worker = self.clone();
                    let db = Arc::clone(db);
                    let identity = identity.clone();
                    tasks.spawn(async move {
                        worker
                            .process_hybrid_frame(&db, id, &identity, required)
                            .await
                    });
                }
                match tasks.join_next().await {
                    None => break,
                    Some(Ok(Ok(true))) => count += 1,
                    Some(Ok(Ok(false))) => {}
                    Some(Ok(Err(error))) => {
                        first_error.get_or_insert(error);
                    }
                    Some(Err(join)) => {
                        first_error.get_or_insert(anyhow::anyhow!("frame task failed: {join}"));
                    }
                }
            }
        }
        if count > 0 {
            let mut status = self.status.lock().await;
            status.redacted_total += count as u64;
            status.last_redacted_at = Some(chrono::Utc::now());
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(count),
        }
    }

    /// One consistent read of a frame's payload and its window/url columns;
    /// `None` when the frame is gone.
    async fn read_hybrid_frame(
        db: &screenpipe_db::DatabaseManager,
        id: i64,
    ) -> Result<Option<(FramePayload, Option<String>, Option<String>)>, sqlx::Error> {
        let token = db.storage_read_token().await?;
        let Some(payload) = db.frame_payloads(&[id], Projection::All).await?.remove(&id) else {
            return Ok(None);
        };
        let row = sqlx::query("SELECT window_name,browser_url FROM frames WHERE id=?")
            .bind(id)
            .fetch_one(&db.pool)
            .await?;
        let window: Option<String> = row.try_get("window_name")?;
        let url: Option<String> = row.try_get("browser_url")?;
        {
            let _admission = token.admit(&db.pool).await?;
        }
        Ok(Some((payload, window, url)))
    }

    /// Redact one frame's configured surfaces and write them back. `Ok(true)`
    /// when the frame was rewritten; a detection failure schedules a retry and
    /// returns `Ok(false)`; database errors are `Err`.
    async fn process_hybrid_frame(
        &self,
        db: &screenpipe_db::DatabaseManager,
        id: i64,
        identity: &str,
        required: u8,
    ) -> Result<bool, anyhow::Error> {
        let complete = CompleteRedactor(self.redactor.as_ref());
        let columns = self.cfg.columns;
        // Every frame write (capture, sealing, and — while draining — the
        // other frame tasks) bumps the storage revision and invalidates a
        // read in flight. Retry the read a few times rather than failing the
        // whole batch into the worker's 2 s error backoff.
        let mut attempt = 0u64;
        let (mut payload, mut window, mut url) = loop {
            match Self::read_hybrid_frame(db, id).await {
                Ok(Some(read)) => break read,
                Ok(None) => return Ok(false),
                Err(error) if is_revision_change(&error) && attempt < READ_ATTEMPTS => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(10 * attempt)).await;
                }
                Err(error) => return Err(error.into()),
            }
        };
        let detection = async {
            if let Some(text) = payload.full_text.as_mut().filter(|s| !s.is_empty()) {
                *text = complete.redact(text).await?.redacted;
            }
            if columns.accessibility_text {
                if let Some(text) = payload
                    .accessibility_text
                    .as_mut()
                    .filter(|s| !s.is_empty())
                {
                    *text = complete.redact(text).await?.redacted;
                }
            }
            if columns.accessibility_tree {
                if let Some(json) = payload
                    .accessibility_tree_json
                    .as_mut()
                    .filter(|s| !s.is_empty())
                {
                    if let Some(clean) = crate::tree_json::redact_tree_json_with_redactor_fields(
                        json,
                        &complete,
                        &columns.a11y_json_fields(),
                    )
                    .await?
                    {
                        *json = clean;
                    }
                }
            }
            if let Some(json) = payload.text_json.as_mut().filter(|s| !s.is_empty()) {
                if let Some(clean) =
                    crate::ocr_json::redact_ocr_text_json_with_redactor(json, &complete).await?
                {
                    *json = clean;
                }
            }
            if columns.window_name {
                if let Some(text) = window.as_mut().filter(|s| !s.is_empty()) {
                    *text = complete.redact(text).await?.redacted;
                }
            }
            if columns.browser_url {
                if let Some(text) = url.as_mut().filter(|s| !s.is_empty()) {
                    *text = complete.redact(text).await?.redacted;
                }
            }
            Ok::<_, anyhow::Error>(())
        }
        .await;
        match detection {
            Ok(()) => Ok(db
                .replace_frame_payload(
                    &payload,
                    identity,
                    required,
                    window.as_deref(),
                    url.as_deref(),
                )
                .await?),
            Err(error) => {
                // The receipt contains an error class, never captured text
                // or detector output. Retry stays scoped to this generation.
                let writer = self.writer.lock().await?;
                sqlx::query("UPDATE frame_payloads SET attempts=attempts+1,retry_at=strftime('%s','now')+MIN(3600,30*(attempts+1)),last_error='configured surface processing failed' WHERE frame_id=? AND generation=?")
                    .bind(id).bind(payload.generation).execute(writer.pool()).await?;
                let mut status = self.status.lock().await;
                status.last_error = Some("frame surface processing failed; retry scheduled".into());
                tracing::debug!(frame_id=id,error_type=%std::any::type_name_of_val(&error),"frame privacy processing deferred");
                Ok(false)
            }
        }
    }
}
