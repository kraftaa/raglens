use crate::config::Config;
use crate::embeddings::{embed_corpus, Embedder};
use crate::model::{
    ChunkEmbedding, Corpus, RankedChunk, RetrievalExplanation, RetrievalResult,
};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ExplainedChunk {
    pub rank: usize,
    pub chunk_id: String,
    pub doc_id: String,
    pub score: f32,
    pub explanation: RetrievalExplanation,
}

#[derive(Clone, Debug)]
pub struct ExplanationReport {
    pub ranked: Vec<ExplainedChunk>,
}

#[derive(Clone, Debug)]
pub struct CompareReport {
    pub ranked: Vec<ExplainedChunk>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuerySpec {
    pub id: String,
    pub query: String,
    #[serde(default)]
    pub expect_docs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QueriesFile {
    queries: Vec<QuerySpec>,
}

#[derive(Clone, Debug)]
pub struct SimulationOutput {
    pub results: Vec<RetrievalResult>,
    pub queries: Vec<QuerySpec>,
}

pub fn simulate_retrieval(
    corpus: &Corpus,
    embedder: &dyn Embedder,
    queries_file: Option<&Path>,
    config: &Config,
) -> Result<SimulationOutput> {
    let corpus_embeddings = embed_corpus(corpus, embedder)?;
    let chunk_lookup: HashMap<_, _> = corpus
        .chunks
        .iter()
        .map(|c| (c.chunk_id.clone(), c))
        .collect();

    let queries = if let Some(path) = queries_file {
        load_queries(path)?
    } else {
        synthesize_queries(corpus)
    };

    let query_texts: Vec<String> = queries.iter().map(|q| q.query.clone()).collect();
    let query_vectors = embedder.embed_batch(&query_texts)?;

    let mut results = Vec::new();
    for (spec, qvec) in queries.iter().zip(query_vectors.into_iter()) {
        let ranked = score_query(&corpus_embeddings, &chunk_lookup, &qvec, config.top_k);
        results.push(RetrievalResult {
            query_id: spec.id.clone(),
            query_text: spec.query.clone(),
            ranked,
        });
    }

    Ok(SimulationOutput { results, queries })
}

pub fn explain_query(
    corpus: &Corpus,
    embedder: &dyn Embedder,
    query: &str,
    config: &Config,
) -> Result<ExplanationReport> {
    let corpus_embeddings = embed_corpus(corpus, embedder)?;
    let chunk_lookup: HashMap<_, _> = corpus
        .chunks
        .iter()
        .map(|c| (c.chunk_id.clone(), c))
        .collect();

    let query_vec = embedder.embed_batch(&[query.to_string()])?.remove(0);
    let ranked = score_query(&corpus_embeddings, &chunk_lookup, &query_vec, config.top_k);

    let explained = ranked
        .iter()
        .map(|r| {
            let chunk = chunk_lookup.get(&r.chunk_id).unwrap();
            ExplainedChunk {
                rank: r.rank,
                chunk_id: r.chunk_id.clone(),
                doc_id: chunk.doc_id.clone(),
                score: r.score,
                explanation: build_explanation(query, chunk, r.score),
            }
        })
        .collect();

    Ok(ExplanationReport { ranked: explained })
}

pub fn compare_query(
    corpus: &Corpus,
    embedder: &dyn Embedder,
    query: &str,
    config: &Config,
) -> Result<CompareReport> {
    let report = explain_query(corpus, embedder, query, config)?;
    // limit to top 5 for comparison output
    let ranked = report
        .ranked
        .into_iter()
        .take(5)
        .collect();
    Ok(CompareReport { ranked })
}

fn load_queries(path: &Path) -> Result<Vec<QuerySpec>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("reading queries file {}", path.display()))?;
    let qf: QueriesFile =
        serde_yaml::from_str(&content).context("parsing queries YAML")?;
    Ok(qf.queries)
}

fn synthesize_queries(corpus: &Corpus) -> Vec<QuerySpec> {
    corpus
        .documents
        .iter()
        .take(16)
        .enumerate()
        .map(|(i, doc)| QuerySpec {
            id: format!("auto_{}", i),
            query: doc
                .title
                .clone()
                .unwrap_or_else(|| doc.id.clone()),
            expect_docs: vec![doc.id.clone()],
        })
        .collect()
}

fn score_query(
    corpus_embeddings: &[ChunkEmbedding],
    chunk_lookup: &HashMap<String, &crate::model::Chunk>,
    query_vec: &[f32],
    top_k: usize,
) -> Vec<RankedChunk> {
    let mut scores: Vec<RankedChunk> = corpus_embeddings
        .iter()
        .filter_map(|emb| {
            let chunk = chunk_lookup.get(&emb.chunk_id)?;
            let score = cosine_similarity(query_vec, &emb.vector);
            Some(RankedChunk {
                chunk_id: chunk.chunk_id.clone(),
                score,
                rank: 0,
            })
        })
        .collect();

    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scores.truncate(top_k);
    for (idx, item) in scores.iter_mut().enumerate() {
        item.rank = idx + 1;
    }
    scores
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn build_explanation(query: &str, chunk: &crate::model::Chunk, similarity: f32) -> RetrievalExplanation {
    let query_lower = query.to_lowercase();
    let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();
    let chunk_lower = chunk.text.to_lowercase();
    let keyword_overlap = query_tokens
        .iter()
        .filter(|t| chunk_lower.contains(&t[..]))
        .count();
    let phrase_match = chunk_lower.contains(&query_lower);
    RetrievalExplanation {
        chunk_id: chunk.chunk_id.clone(),
        similarity,
        keyword_overlap,
        phrase_match,
        token_count: chunk.token_count,
    }
}
