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
    pub seed: u64,
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
    // Backward-compatible top-level embedding keys used in README/examples.
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    cache_dir: Option<String>,
    low_sim_threshold: Option<f32>,
    no_match_threshold: Option<f32>,
    max_retries: Option<usize>,
    retry_backoff_ms: Option<u64>,
    request_timeout_ms: Option<u64>,
    seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 400,
            chunk_overlap: 40,
            max_tokens: 1200,
            min_tokens: 60,
            top_k: 5,
            dominant_threshold: 0.3,
            required_metadata: vec!["product".into(), "region".into(), "version".into()],
            embedder: EmbedderConfig::Null,
            cache_dir: std::path::PathBuf::from(".rag-audit-cache"),
            low_sim_threshold: 0.35,
            no_match_threshold: 0.25,
            max_retries: 3,
            retry_backoff_ms: 300,
            request_timeout_ms: 15_000,
            seed: 42,
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

        let embedder = if let Some(e) = raw.embedder {
            e
        } else if let Some(provider) = raw.provider {
            match provider.to_lowercase().as_str() {
                "null" => EmbedderConfig::Null,
                "openai" => EmbedderConfig::OpenAI {
                    model: raw
                        .model
                        .clone()
                        .unwrap_or_else(|| "text-embedding-3-small".to_string()),
                    base_url: raw.base_url.clone().unwrap_or_else(default_openai_base),
                },
                other => anyhow::bail!("unknown provider in config: {}", other),
            }
        } else {
            default.embedder
        };

        let cfg = Config {
            chunk_size: raw.chunk_size.unwrap_or(default.chunk_size),
            chunk_overlap: raw.chunk_overlap.unwrap_or(default.chunk_overlap),
            max_tokens: raw.max_tokens.unwrap_or(default.max_tokens),
            min_tokens: raw.min_tokens.unwrap_or(default.min_tokens),
            top_k: raw.top_k.unwrap_or(default.top_k),
            dominant_threshold: raw.dominant_threshold.unwrap_or(default.dominant_threshold),
            required_metadata: raw.required_metadata.unwrap_or(default.required_metadata),
            embedder,
            cache_dir: raw
                .cache_dir
                .map(std::path::PathBuf::from)
                .unwrap_or(default.cache_dir),
            low_sim_threshold: raw.low_sim_threshold.unwrap_or(default.low_sim_threshold),
            no_match_threshold: raw.no_match_threshold.unwrap_or(default.no_match_threshold),
            max_retries: raw.max_retries.unwrap_or(default.max_retries),
            retry_backoff_ms: raw.retry_backoff_ms.unwrap_or(default.retry_backoff_ms),
            request_timeout_ms: raw.request_timeout_ms.unwrap_or(default.request_timeout_ms),
            seed: raw.seed.unwrap_or(default.seed),
        };
        cfg.validate()?;
        Ok(cfg)
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

    pub fn override_thresholds(&mut self, low: Option<f32>, no_match: Option<f32>) {
        if let Some(v) = low {
            self.low_sim_threshold = v;
        }
        if let Some(v) = no_match {
            self.no_match_threshold = v;
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            anyhow::bail!("chunk_size must be > 0");
        }
        if self.top_k == 0 {
            anyhow::bail!("top_k must be > 0");
        }
        if self.chunk_overlap >= self.chunk_size {
            anyhow::bail!(
                "chunk_overlap ({}) must be < chunk_size ({})",
                self.chunk_overlap,
                self.chunk_size
            );
        }
        if self.max_tokens < self.min_tokens {
            anyhow::bail!(
                "max_tokens ({}) must be >= min_tokens ({})",
                self.max_tokens,
                self.min_tokens
            );
        }
        if !(-1.0..=1.0).contains(&self.low_sim_threshold) {
            anyhow::bail!(
                "low_sim_threshold must be in [-1, 1], got {}",
                self.low_sim_threshold
            );
        }
        if !(-1.0..=1.0).contains(&self.no_match_threshold) {
            anyhow::bail!(
                "no_match_threshold must be in [-1, 1], got {}",
                self.no_match_threshold
            );
        }
        if self.no_match_threshold > self.low_sim_threshold {
            anyhow::bail!(
                "no_match_threshold ({}) must be <= low_sim_threshold ({})",
                self.no_match_threshold,
                self.low_sim_threshold
            );
        }
        if !(0.0..=1.0).contains(&self.dominant_threshold) {
            anyhow::bail!(
                "dominant_threshold must be in [0, 1], got {}",
                self.dominant_threshold
            );
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

fn default_openai_base() -> String {
    "https://api.openai.com/v1".to_string()
}

#[cfg(test)]
mod tests {
    use super::Config;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn invalid_threshold_order_fails_validation() {
        let cfg = Config {
            low_sim_threshold: 0.2,
            no_match_threshold: 0.4,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_dominant_threshold_fails_validation() {
        let cfg = Config {
            dominant_threshold: 1.2,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_chunking_parameters_fail_validation() {
        let bad_chunk_size = Config {
            chunk_size: 0,
            ..Config::default()
        };
        assert!(bad_chunk_size.validate().is_err());

        let bad_overlap = Config {
            chunk_size: 100,
            chunk_overlap: 100,
            ..Config::default()
        };
        assert!(bad_overlap.validate().is_err());

        let bad_token_limits = Config {
            min_tokens: 200,
            max_tokens: 100,
            ..Config::default()
        };
        assert!(bad_token_limits.validate().is_err());

        let bad_top_k = Config {
            top_k: 0,
            ..Config::default()
        };
        assert!(bad_top_k.validate().is_err());
    }

    #[test]
    fn supports_top_level_provider_style_config() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rag_audit_cfg_{stamp}.toml"));
        fs::write(
            &path,
            r#"
provider = "openai"
model = "text-embedding-3-small"
base_url = "https://api.openai.com/v1"
"#,
        )
        .unwrap();

        let cfg = Config::load(Some(&path)).expect("config should parse");
        match cfg.embedder {
            super::EmbedderConfig::OpenAI { model, base_url } => {
                assert_eq!(model, "text-embedding-3-small");
                assert_eq!(base_url, "https://api.openai.com/v1");
            }
            _ => panic!("expected openai embedder"),
        }

        let _ = fs::remove_file(path);
    }
}
