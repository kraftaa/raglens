use crate::config::{Config, EmbedderConfig};
use crate::model::{ChunkEmbedding, Corpus};
use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

pub trait Embedder: Sync + Send {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
}

#[derive(Clone)]
pub struct NullEmbedder {
    dim: usize,
}

impl Default for NullEmbedder {
    fn default() -> Self {
        Self { dim: 64 }
    }
}

impl NullEmbedder {
    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0f32; self.dim];
        for token in text.split_whitespace() {
            let idx = self.hash_token(token) % self.dim as u64;
            vec[idx as usize] += 1.0;
        }
        vec
    }

    fn hash_token(&self, token: &str) -> u64 {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        h.finish()
    }
}

impl Embedder for NullEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[allow(dead_code)]
pub fn embed_corpus(corpus: &Corpus, embedder: &dyn Embedder) -> Result<Vec<ChunkEmbedding>> {
    let texts: Vec<String> = corpus.chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed_batch(&texts)?;

    Ok(corpus
        .chunks
        .iter()
        .zip(embeddings)
        .map(|(chunk, vector)| ChunkEmbedding {
            chunk_id: chunk.chunk_id.clone(),
            vector,
        })
        .collect())
}

/// Simple on-disk cache to avoid re-embedding unchanged chunks.
pub struct EmbeddingCache {
    path: PathBuf,
    map: HashMap<String, Vec<f32>>,
}

impl EmbeddingCache {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Ok(Self {
                path,
                map: HashMap::new(),
            });
        }
        let data = fs::read(&path).context("reading embedding cache")?;
        let map: HashMap<String, Vec<f32>> =
            serde_json::from_slice(&data).context("parsing embedding cache json")?;
        Ok(Self { path, map })
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.map.get(key)
    }

    pub fn insert(&mut self, key: String, vector: Vec<f32>) {
        self.map.insert(key, vector);
    }

    pub fn persist(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).ok();
        }
        let data = serde_json::to_vec_pretty(&self.map)?;
        fs::write(&self.path, data).context("writing embedding cache")
    }
}

pub struct OpenAIEmbedder {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    dim_hint: usize,
    batch_size: usize,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String, model: String, base_url: String, timeout_ms: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .build()?;
        // we cannot know dim without calling; leave hint 1536 typical
        Ok(Self {
            client,
            api_key,
            model,
            base_url,
            dim_hint: 1536,
            batch_size: 64,
        })
    }

    fn batch_inputs<'a>(&self, texts: &'a [String]) -> Vec<&'a [String]> {
        let mut batches = Vec::new();
        let mut start = 0;
        while start < texts.len() {
            let mut end = start;
            let mut tokens = 0usize;
            while end < texts.len() && (end - start) < self.batch_size && tokens < 2000 {
                tokens += rough_tokens(&texts[end]);
                end += 1;
            }
            batches.push(&texts[start..end]);
            start = end;
        }
        batches
    }
}

impl Embedder for OpenAIEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let mut out = Vec::with_capacity(texts.len());
        for chunk_group in self.batch_inputs(texts) {
            let body = serde_json::json!({
                "model": self.model,
                "input": chunk_group,
            });
            let mut last_err = None;
            for attempt in 0..=2 {
                let resp = self
                    .client
                    .post(&url)
                    .bearer_auth(&self.api_key)
                    .json(&body)
                    .send();
                match resp {
                    Ok(r) => {
                        let r = r
                            .error_for_status()
                            .context("openai embedding error status")?;
                        let parsed: EmbeddingResponse =
                            r.json().context("parsing openai embedding response")?;
                        if parsed.data.len() != chunk_group.len() {
                            return Err(anyhow!(
                                "openai embedding count mismatch: expected {}, got {}",
                                chunk_group.len(),
                                parsed.data.len()
                            ));
                        }
                        for item in parsed.data {
                            out.push(item.embedding);
                        }
                        last_err = None;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        std::thread::sleep(std::time::Duration::from_millis(
                            300 * (attempt + 1) as u64,
                        ));
                    }
                }
            }
            if let Some(err) = last_err {
                return Err(anyhow!(
                    "openai embedding request failed after retries: {}",
                    err
                ));
            }
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim_hint
    }
}

pub fn build_embedder(config: &Config) -> Result<Box<dyn Embedder>> {
    match &config.embedder {
        EmbedderConfig::Null => Ok(Box::new(NullEmbedder::default())),
        EmbedderConfig::OpenAI { model, base_url } => {
            let api_key = std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY not set for openai embedder")?;
            Ok(Box::new(OpenAIEmbedder::new(
                api_key,
                model.clone(),
                base_url.clone(),
                config.request_timeout_ms,
            )?))
        }
    }
}

pub fn hash_text_key(text: &str, dim: usize, model: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update(dim.to_le_bytes());
    hasher.update(model.as_bytes());
    hex::encode(hasher.finalize())
}

fn rough_tokens(text: &str) -> usize {
    // heuristic: 1 token per 0.75 words
    ((text.split_whitespace().count() as f32) * 1.33).ceil() as usize
}
