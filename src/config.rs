use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub top_k: usize,
    pub dominant_threshold: f32,
    pub required_metadata: Vec<String>,
    pub embedder: EmbedderConfig,
    pub cache_dir: std::path::PathBuf,
    pub low_sim_threshold: f32,
    pub no_match_threshold: f32,
    pub max_retries: usize,
    pub retry_backoff_ms: u64,
    pub request_timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum EmbedderConfig {
    Null,
    OpenAI {
        model: String,
        #[serde(default = "default_openai_base")]
        base_url: String,
    },
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
    top_k: Option<usize>,
    dominant_threshold: Option<f32>,
    required_metadata: Option<Vec<String>>,
    embedder: Option<EmbedderConfig>,
    cache_dir: Option<String>,
    low_sim_threshold: Option<f32>,
    no_match_threshold: Option<f32>,
    max_retries: Option<usize>,
    retry_backoff_ms: Option<u64>,
    request_timeout_ms: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 400,
            chunk_overlap: 40,
            max_tokens: 1200,
            min_tokens: 60,
            top_k: 5,
            dominant_threshold: 0.2,
            required_metadata: vec!["product".into(), "region".into(), "version".into()],
            embedder: EmbedderConfig::Null,
            cache_dir: std::path::PathBuf::from(".rag-audit-cache"),
            low_sim_threshold: 0.35,
            no_match_threshold: 0.25,
            max_retries: 3,
            retry_backoff_ms: 300,
            request_timeout_ms: 15_000,
        }
    }
}

impl Config {
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let default = Config::default();
        let path = match explicit {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from("rag-audit.toml"),
        };

        if !path.exists() {
            return Ok(default);
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let raw: RawConfig = toml::from_str(&content).context("parsing rag-audit.toml")?;

        Ok(Config {
            chunk_size: raw.chunk_size.unwrap_or(default.chunk_size),
            chunk_overlap: raw.chunk_overlap.unwrap_or(default.chunk_overlap),
            max_tokens: raw.max_tokens.unwrap_or(default.max_tokens),
            min_tokens: raw.min_tokens.unwrap_or(default.min_tokens),
            top_k: raw.top_k.unwrap_or(default.top_k),
            dominant_threshold: raw.dominant_threshold.unwrap_or(default.dominant_threshold),
            required_metadata: raw.required_metadata.unwrap_or(default.required_metadata),
            embedder: raw.embedder.unwrap_or(default.embedder),
            cache_dir: raw
                .cache_dir
                .map(std::path::PathBuf::from)
                .unwrap_or(default.cache_dir),
            low_sim_threshold: raw.low_sim_threshold.unwrap_or(default.low_sim_threshold),
            no_match_threshold: raw.no_match_threshold.unwrap_or(default.no_match_threshold),
            max_retries: raw.max_retries.unwrap_or(default.max_retries),
            retry_backoff_ms: raw.retry_backoff_ms.unwrap_or(default.retry_backoff_ms),
            request_timeout_ms: raw.request_timeout_ms.unwrap_or(default.request_timeout_ms),
        })
    }

    pub fn override_embedder(&mut self, value: Option<&str>) -> Result<()> {
        if let Some(v) = value {
            match v.to_lowercase().as_str() {
                "null" => self.embedder = EmbedderConfig::Null,
                "openai" => {
                    // keep existing model/base if already openai, else set default
                    let model = match &self.embedder {
                        EmbedderConfig::OpenAI { model, .. } => model.clone(),
                        _ => "text-embedding-3-small".to_string(),
                    };
                    let base = match &self.embedder {
                        EmbedderConfig::OpenAI { base_url, .. } => base_url.clone(),
                        _ => default_openai_base(),
                    };
                    self.embedder = EmbedderConfig::OpenAI {
                        model,
                        base_url: base,
                    };
                }
                other => return Err(anyhow::anyhow!("unknown embedder override: {}", other)),
            }
        }
        Ok(())
    }
}

fn default_openai_base() -> String {
    "https://api.openai.com/v1".to_string()
}
