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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalExplanation {
    pub chunk_id: String,
    pub similarity: f32,
    pub keyword_overlap: usize,
    pub phrase_match: bool,
    pub token_count: usize,
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
