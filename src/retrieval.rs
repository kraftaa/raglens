use crate::config::Config;
use crate::embeddings::{hash_text_key, Embedder, EmbeddingCache};
use crate::model::{ChunkEmbedding, Corpus, RankedChunk, RetrievalExplanation, RetrievalResult};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct ExplainedChunk {
    pub rank: usize,
    pub chunk_id: String,
    pub doc_id: String,
    pub score: f32,
    pub explanation: RetrievalExplanation,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExplanationReport {
    pub ranked: Vec<ExplainedChunk>,
}

#[derive(Clone, Debug, Serialize)]
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
    let corpus_embeddings = embed_with_cache(corpus, embedder, config)?;
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
    for (spec, qvec) in queries.iter().zip(query_vectors) {
        let ranked = score_query(
            &corpus_embeddings,
            &chunk_lookup,
            &spec.query,
            &qvec,
            config.top_k,
        );
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
    let corpus_embeddings = embed_with_cache(corpus, embedder, config)?;
    let chunk_lookup: HashMap<_, _> = corpus
        .chunks
        .iter()
        .map(|c| (c.chunk_id.clone(), c))
        .collect();

    let query_vec = embedder.embed_batch(&[query.to_string()])?.remove(0);
    let ranked = score_query(
        &corpus_embeddings,
        &chunk_lookup,
        query,
        &query_vec,
        config.top_k,
    );
    let emb_lookup: HashMap<&str, &[f32]> = corpus_embeddings
        .iter()
        .map(|e| (e.chunk_id.as_str(), e.vector.as_slice()))
        .collect();

    let explained = ranked
        .iter()
        .filter_map(|r| {
            let chunk = chunk_lookup.get(&r.chunk_id)?;
            let semantic = emb_lookup
                .get(r.chunk_id.as_str())
                .map(|vec| cosine_similarity(&query_vec, vec))
                .unwrap_or(0.0);
            Some(ExplainedChunk {
                rank: r.rank,
                chunk_id: r.chunk_id.clone(),
                doc_id: chunk.doc_id.clone(),
                score: r.score,
                explanation: build_explanation(query, chunk, semantic),
            })
        })
        .collect();

    Ok(ExplanationReport { ranked: explained })
}

fn load_queries(path: &Path) -> Result<Vec<QuerySpec>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("reading queries file {}", path.display()))?;
    let is_yaml_ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "yaml" | "yml"))
        .unwrap_or(false);
    let looks_like_yaml_queries = content.trim_start().starts_with("queries:");

    if is_yaml_ext || looks_like_yaml_queries {
        let qf: QueriesFile = serde_yaml::from_str(&content)
            .with_context(|| format!("parsing YAML queries in {}", path.display()))?;
        return Ok(qf.queries);
    }

    if let Ok(qf) = serde_yaml::from_str::<QueriesFile>(&content) {
        return Ok(qf.queries);
    }
    // fallback: plain text, one query per non-empty line.
    // Optional structured format per line:
    // id<TAB>query<TAB>expect_doc1,expect_doc2
    let mut out = Vec::new();
    let mut line_query_index = 0usize;
    for line in content.lines() {
        let q = line.trim();
        if q.is_empty() || q.starts_with('#') {
            continue;
        }
        line_query_index += 1;
        let parts: Vec<&str> = q.splitn(3, '\t').collect();
        if parts.len() >= 2 {
            let id = if parts[0].trim().is_empty() {
                format!("q{}", line_query_index)
            } else {
                parts[0].trim().to_string()
            };
            let query = parts[1].trim().to_string();
            let expect_docs = if parts.len() == 3 {
                parts[2]
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            } else {
                Vec::new()
            };
            out.push(QuerySpec {
                id,
                query,
                expect_docs,
            });
            continue;
        }
        out.push(QuerySpec {
            id: format!("q{}", line_query_index),
            query: q.to_string(),
            expect_docs: Vec::new(),
        });
    }
    Ok(out)
}

fn synthesize_queries(corpus: &Corpus) -> Vec<QuerySpec> {
    corpus
        .documents
        .iter()
        .take(16)
        .enumerate()
        .map(|(i, doc)| QuerySpec {
            id: format!("auto_{}", i),
            query: doc.title.clone().unwrap_or_else(|| doc.id.clone()),
            expect_docs: vec![doc.id.clone()],
        })
        .collect()
}

fn score_query(
    corpus_embeddings: &[ChunkEmbedding],
    chunk_lookup: &HashMap<String, &crate::model::Chunk>,
    query: &str,
    query_vec: &[f32],
    top_k: usize,
) -> Vec<RankedChunk> {
    let mut scores: Vec<RankedChunk> = corpus_embeddings
        .iter()
        .filter_map(|emb| {
            let chunk = chunk_lookup.get(&emb.chunk_id)?;
            let raw_score = cosine_similarity(query_vec, &emb.vector);
            let semantic = if raw_score.is_finite() {
                raw_score
            } else {
                0.0
            };
            let explanation = build_explanation(query, chunk, semantic);
            let score = if explanation.total_score.is_finite() {
                explanation.total_score
            } else {
                semantic
            };
            Some(RankedChunk {
                chunk_id: chunk.chunk_id.clone(),
                score,
                rank: 0,
            })
        })
        .collect();

    scores.sort_by(|a, b| b.score.total_cmp(&a.score));
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
    let sim = dot / (na.sqrt() * nb.sqrt());
    if sim.is_finite() {
        sim
    } else {
        0.0
    }
}

fn build_explanation(
    query: &str,
    chunk: &crate::model::Chunk,
    similarity: f32,
) -> RetrievalExplanation {
    let query_lower = query.to_lowercase();
    let chunk_lower = chunk.text.to_lowercase();
    let query_tokens = tokenize_terms(&query_lower);
    let chunk_tokens = tokenize_terms(&chunk_lower);
    let keyword_overlap = query_tokens.intersection(&chunk_tokens).count();
    let phrase_match = chunk_lower.contains(&query_lower);
    let keyword_overlap_norm = keyword_overlap as f32 / (query_tokens.len().max(1) as f32);
    let keyword_boost = (keyword_overlap_norm * 0.15).min(0.15);
    let phrase_boost = if phrase_match { 0.08 } else { 0.0 };
    let length_penalty = if chunk.token_count > 800 {
        ((chunk.token_count as f32 - 800.0) / 800.0).min(0.3)
    } else {
        0.0
    };
    let metadata_boost = 0.0; // placeholder for filter-aware scoring
    let total_score = similarity + keyword_boost + phrase_boost + metadata_boost - length_penalty;
    RetrievalExplanation {
        chunk_id: chunk.chunk_id.clone(),
        similarity,
        keyword_overlap,
        keyword_overlap_norm,
        keyword_boost,
        phrase_match,
        phrase_boost,
        token_count: chunk.token_count,
        metadata_boost,
        length_penalty,
        total_score,
    }
}

fn tokenize_terms(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

fn embed_with_cache(
    corpus: &Corpus,
    embedder: &dyn Embedder,
    config: &Config,
) -> Result<Vec<ChunkEmbedding>> {
    let mut cache = EmbeddingCache::load(config.cache_dir.join("embeddings.json"))?;
    let model_tag = match &config.embedder {
        crate::config::EmbedderConfig::Null => "null",
        crate::config::EmbedderConfig::OpenAI { model, .. } => model.as_str(),
    };

    let mut vectors = Vec::with_capacity(corpus.chunks.len());
    let mut to_embed = Vec::new();
    for chunk in &corpus.chunks {
        let key = hash_text_key(&chunk.text, embedder.dim(), model_tag);
        if let Some(vec) = cache.get(&key) {
            vectors.push(ChunkEmbedding {
                chunk_id: chunk.chunk_id.clone(),
                vector: vec.clone(),
            });
        } else {
            to_embed.push((key, chunk));
        }
    }

    if !to_embed.is_empty() {
        let texts: Vec<String> = to_embed.iter().map(|(_, c)| c.text.clone()).collect();
        let embedded = embedder.embed_batch(&texts)?;
        for ((key, chunk), vector) in to_embed.into_iter().zip(embedded) {
            cache.insert(key, vector.clone());
            vectors.push(ChunkEmbedding {
                chunk_id: chunk.chunk_id.clone(),
                vector,
            });
        }
        if let Err(err) = cache.persist() {
            eprintln!("warning: embedding cache persist failed: {err}");
        }
    }

    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::load_queries;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn load_queries_supports_tab_separated_format() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rag_audit_queries_{stamp}.txt"));
        let data = "# comment\nrefund_window\trefund after 90 days\trefund_policy.md,faq.md\nshipping\tshipping delay\nplain fallback query\n";
        fs::write(&path, data).unwrap();

        let parsed = load_queries(&path).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].id, "refund_window");
        assert_eq!(parsed[0].query, "refund after 90 days");
        assert_eq!(
            parsed[0].expect_docs,
            vec!["refund_policy.md".to_string(), "faq.md".to_string()]
        );
        assert_eq!(parsed[1].id, "shipping");
        assert_eq!(parsed[1].query, "shipping delay");
        assert_eq!(parsed[2].id, "q3");
        assert_eq!(parsed[2].query, "plain fallback query");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_queries_fails_fast_for_invalid_yaml_files() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rag_audit_queries_bad_{stamp}.yaml"));
        let data = "queries:\n  - id: q1\n    query: ok\n  - id q2\n    query: broken\n";
        fs::write(&path, data).unwrap();

        let err = load_queries(&path).expect_err("invalid yaml should return an error");
        assert!(err.to_string().contains("parsing YAML queries"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_queries_ignores_blank_and_comment_lines() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rag_audit_queries_comment_{stamp}.txt"));
        let data = "\n# comment 1\n\nfirst query\n# comment 2\nsecond query\n";
        fs::write(&path, data).unwrap();

        let parsed = load_queries(&path).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "q1");
        assert_eq!(parsed[0].query, "first query");
        assert_eq!(parsed[1].id, "q2");
        assert_eq!(parsed[1].query, "second query");

        let _ = fs::remove_file(path);
    }
}
