use crate::config::Config;
use crate::model::{Corpus, Finding, RetrievalResult, Severity};
use crate::retrieval::QuerySpec;
use indexmap::IndexMap;
use serde_json::json;

pub fn run_readiness(corpus: &Corpus, config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    findings.extend(analyze_chunks(corpus, config));
    findings.extend(analyze_metadata(corpus, config));
    findings
}

pub fn analyze_retrieval(
    retrievals: &[RetrievalResult],
    queries: &[QuerySpec],
    config: &Config,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    // Dominant documents: count document frequency across results
    let mut freq: IndexMap<String, usize> = IndexMap::new();
    for result in retrievals {
        for ranked in &result.ranked {
            let doc_id = chunk_doc_id(&ranked.chunk_id);
            *freq.entry(doc_id).or_insert(0) += 1;
        }
    }

    let total_queries = retrievals.len().max(1);
    for (doc, count) in freq.iter() {
        let rate = *count as f32 / total_queries as f32;
        if rate >= config.dominant_threshold {
            findings.push(Finding {
                severity: Severity::High,
                code: "DOMINANT_DOCUMENT".to_string(),
                message: format!("{doc} retrieved in {:.0}%", rate * 100.0),
                data: IndexMap::from([
                    ("doc".into(), json!(doc)),
                    ("rate".into(), json!(rate)),
                ]),
            });
        }
    }

    // Expectation failures
    for spec in queries {
        if spec.expect_docs.is_empty() {
            continue;
        }
        if let Some(result) = retrievals.iter().find(|r| r.query_id == spec.id) {
            let mut best_rank = None;
            for doc in &spec.expect_docs {
                if let Some(hit) = result
                    .ranked
                    .iter()
                    .find(|r| chunk_doc_id(&r.chunk_id) == *doc)
                {
                    best_rank = Some(hit.rank);
                    break;
                }
            }
            if best_rank.is_none() || best_rank.unwrap() > config.top_k {
                findings.push(Finding {
                    severity: Severity::Fail,
                    code: "EXPECTATION_MISS".into(),
                    message: format!(
                        "Query {} expected {:?}, not found in top {}",
                        spec.id, spec.expect_docs, config.top_k
                    ),
                    data: IndexMap::from([
                        ("query_id".into(), json!(spec.id)),
                        ("expected".into(), json!(spec.expect_docs)),
                    ]),
                });
            }
        }
    }

    findings
}

pub fn analyze_chunks(corpus: &Corpus, config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    let oversized: Vec<_> = corpus
        .chunks
        .iter()
        .filter(|c| c.token_count > config.max_tokens)
        .collect();
    let tiny: Vec<_> = corpus
        .chunks
        .iter()
        .filter(|c| c.token_count < config.min_tokens)
        .collect();

    if !oversized.is_empty() {
        findings.push(Finding {
            severity: Severity::Medium,
            code: "OVERSIZED_CHUNK".into(),
            message: format!("{} chunks exceed max_tokens {}", oversized.len(), config.max_tokens),
            data: IndexMap::new(),
        });
    }

    if !tiny.is_empty() {
        findings.push(Finding {
            severity: Severity::Low,
            code: "TINY_CHUNK".into(),
            message: format!("{} chunks below min_tokens {}", tiny.len(), config.min_tokens),
            data: IndexMap::new(),
        });
    }

    findings
}

pub fn analyze_topics(corpus: &Corpus, _config: &Config) -> Vec<Finding> {
    let mut counts: IndexMap<String, usize> = IndexMap::new();
    for chunk in &corpus.chunks {
        let topic = chunk
            .heading_path
            .first()
            .cloned()
            .unwrap_or_else(|| "general".to_string());
        *counts.entry(topic).or_insert(0) += 1;
    }

    let total: usize = counts.values().sum();
    let mut findings = Vec::new();
    for (topic, count) in counts.iter() {
        let share = *count as f32 / total.max(1) as f32;
        if share > 0.4 {
            findings.push(Finding {
                severity: Severity::Medium,
                code: "TOPIC_DOMINANCE".into(),
                message: format!("{topic} topic dominates corpus ({:.0}%)", share * 100.0),
                data: IndexMap::new(),
            });
        } else if share < 0.05 {
            findings.push(Finding {
                severity: Severity::Low,
                code: "TOPIC_LOW_COVERAGE".into(),
                message: format!("{topic} coverage very low ({:.0}%)", share * 100.0),
                data: IndexMap::new(),
            });
        }
    }
    findings
}

pub fn analyze_metadata(corpus: &Corpus, config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    for key in &config.required_metadata {
        let missing = corpus
            .documents
            .iter()
            .filter(|d| !d.metadata.contains_key(key))
            .count();
        if missing > 0 {
            findings.push(Finding {
                severity: Severity::Low,
                code: "MISSING_METADATA".into(),
                message: format!("{missing} docs missing {key}"),
                data: IndexMap::new(),
            });
        }
    }
    findings
}

fn chunk_doc_id(chunk_id: &str) -> String {
    chunk_id.split('#').next().unwrap_or(chunk_id).to_string()
}
