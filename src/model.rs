use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub path: PathBuf,
    pub title: Option<String>,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub doc_id: String,
    pub text: String,
    pub token_count: usize,
    pub heading_path: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Corpus {
    pub documents: Vec<Document>,
    pub chunks: Vec<Chunk>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkEmbedding {
    pub chunk_id: String,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedChunk {
    pub chunk_id: String,
    pub score: f32,
    pub rank: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub query_id: String,
    pub query_text: String,
    pub ranked: Vec<RankedChunk>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CoverageSummary {
    pub queries: usize,
    pub good: usize,
    pub weak: usize,
    pub none: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ChunkStats {
    pub documents: usize,
    pub chunks: usize,
    pub avg_tokens: f32,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub p50_tokens: usize,
    pub p95_tokens: usize,
    pub large_chunks: usize,
    pub small_chunks: usize,
    pub duplicate_chunks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SimSummary {
    pub queries: usize,
    pub avg_top1_similarity: f32,
    pub low_similarity_queries: usize,
    pub no_match_queries: usize,
    pub top1_freq: Vec<DocFreq>,
    pub top3_freq: Vec<DocFreq>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocFreq {
    pub doc_id: String,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalExplanation {
    pub chunk_id: String,
    pub similarity: f32,
    pub keyword_overlap: usize,
    pub keyword_overlap_norm: f32,
    pub phrase_match: bool,
    pub token_count: usize,
    pub metadata_boost: f32,
    pub length_penalty: f32,
    pub total_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub data: IndexMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Severity {
    High,
    Medium,
    Low,
    Info,
    Fail,
}
