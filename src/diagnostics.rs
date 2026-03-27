use crate::config::Config;
use crate::model::{ChunkStats, Corpus, DocFreq, Finding, RetrievalResult, Severity, SimSummary};
use crate::retrieval::QuerySpec;
use indexmap::IndexMap;
use serde_json::json;

pub fn run_readiness(corpus: &Corpus, config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    findings.extend(analyze_metadata(corpus, config));
    findings
}

pub fn analyze_retrieval(
    retrievals: &[RetrievalResult],
    queries: &[QuerySpec],
    config: &Config,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    findings.extend(analyze_dominant_docs(retrievals, config));

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

fn analyze_dominant_docs(retrievals: &[RetrievalResult], config: &Config) -> Vec<Finding> {
    let mut findings = Vec::new();
    if retrievals.is_empty() {
        return findings;
    }
    let mut freq_topk: IndexMap<String, usize> = IndexMap::new();
    let mut freq_top1: IndexMap<String, usize> = IndexMap::new();
    for result in retrievals {
        if let Some(top) = result.ranked.first() {
            let doc_id = chunk_doc_id(&top.chunk_id);
            *freq_top1.entry(doc_id).or_insert(0) += 1;
        }
        for ranked in &result.ranked {
            let doc_id = chunk_doc_id(&ranked.chunk_id);
            *freq_topk.entry(doc_id).or_insert(0) += 1;
        }
    }
    let total_queries = retrievals.len().max(1);
    for (doc, count) in freq_topk.iter() {
        let rate = *count as f32 / total_queries as f32;
        if rate >= config.dominant_threshold {
            let top1_rate = freq_top1
                .get(doc)
                .map(|c| *c as f32 / total_queries as f32)
                .unwrap_or(0.0);
            findings.push(Finding {
                severity: Severity::High,
                code: "DOMINANT_DOCUMENT".to_string(),
                message: format!(
                    "{doc} retrieved in {:.0}% of queries (top-k), {:.0}% top-1",
                    rate * 100.0,
                    top1_rate * 100.0
                ),
                data: IndexMap::from([
                    ("doc".into(), json!(doc)),
                    ("rate_topk".into(), json!(rate)),
                    ("rate_top1".into(), json!(top1_rate)),
                ]),
            });
        }
    }
    findings
}

pub fn max_dominant_rate(retrievals: &[RetrievalResult], _config: &Config) -> Option<f32> {
    if retrievals.is_empty() {
        return None;
    }
    let mut freq: IndexMap<String, usize> = IndexMap::new();
    for result in retrievals {
        if let Some(top) = result.ranked.first() {
            let doc_id = chunk_doc_id(&top.chunk_id);
            *freq.entry(doc_id).or_insert(0) += 1;
        }
    }
    let total = retrievals.len() as f32;
    freq.values()
        .map(|c| *c as f32 / total)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
}

pub fn coverage_summary(results: &[RetrievalResult]) -> crate::model::CoverageSummary {
    let mut summary = crate::model::CoverageSummary::default();
    summary.queries = results.len();
    for r in results {
        let top_score = r.ranked.first().map(|c| c.score).unwrap_or(0.0);
        if top_score >= 0.6 {
            summary.good += 1;
        } else if top_score >= 0.3 {
            summary.weak += 1;
        } else {
            summary.none += 1;
        }
    }
    summary
}

pub fn simulate_summary(
    retrievals: &[RetrievalResult],
    low_threshold: f32,
    no_match_threshold: f32,
) -> SimSummary {
    let mut summary = SimSummary::default();
    summary.queries = retrievals.len();

    let mut top1_freq: IndexMap<String, usize> = IndexMap::new();
    let mut top3_freq: IndexMap<String, usize> = IndexMap::new();
    for r in retrievals {
        if let Some(top) = r.ranked.first() {
            *top1_freq.entry(chunk_doc_id(&top.chunk_id)).or_insert(0) += 1;
            summary.avg_top1_similarity += top.score;
            if top.score < low_threshold {
                summary.low_similarity_queries += 1;
                summary.low_similarity_query_ids.push(r.query_id.clone());
            }
            if top.score < no_match_threshold {
                summary.no_match_queries += 1;
                summary.no_match_query_ids.push(r.query_id.clone());
            }
        }
        for item in r.ranked.iter().take(3) {
            *top3_freq.entry(chunk_doc_id(&item.chunk_id)).or_insert(0) += 1;
        }
    }
    if summary.queries > 0 {
        summary.avg_top1_similarity /= summary.queries as f32;
    }
    summary.top1_freq = top1_freq
        .into_iter()
        .map(|(doc_id, count)| DocFreq { doc_id, count })
        .collect();
    summary.top3_freq = top3_freq
        .into_iter()
        .map(|(doc_id, count)| DocFreq { doc_id, count })
        .collect();
    summary
}

pub fn chunk_stats(corpus: &Corpus, config: &Config) -> (ChunkStats, Vec<Finding>) {
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
            message: format!(
                "{} chunks exceed max_tokens {}",
                oversized.len(),
                config.max_tokens
            ),
            data: IndexMap::new(),
        });
    }

    if !tiny.is_empty() {
        findings.push(Finding {
            severity: Severity::Low,
            code: "TINY_CHUNK".into(),
            message: format!(
                "{} chunks below min_tokens {}",
                tiny.len(),
                config.min_tokens
            ),
            data: IndexMap::new(),
        });
    }

    // Simple coherence heuristic: multiple headings in one chunk suggests multi-topic
    let multi_heading: Vec<_> = corpus
        .chunks
        .iter()
        .filter(|c| count_headings(&c.text) >= 3 && c.token_count > config.min_tokens)
        .collect();
    if !multi_heading.is_empty() {
        findings.push(Finding {
            severity: Severity::Medium,
            code: "MULTI_TOPIC_CHUNK".into(),
            message: format!(
                "{} chunks contain 3+ headings (possible multi-topic)",
                multi_heading.len()
            ),
            data: IndexMap::new(),
        });
    }

    // Chunks ending mid-sentence
    let cut_chunks: Vec<_> = corpus
        .chunks
        .iter()
        .filter(|c| c.token_count > config.min_tokens && ends_mid_sentence(&c.text))
        .collect();
    if !cut_chunks.is_empty() {
        findings.push(Finding {
            severity: Severity::Low,
            code: "SENTENCE_CUT".into(),
            message: format!(
                "{} chunks end mid-sentence (consider overlap/tokens)",
                cut_chunks.len()
            ),
            data: IndexMap::new(),
        });
    }

    // Overlap variance heuristic
    let overlaps = compute_overlaps(corpus);
    if !overlaps.is_empty() {
        let mean: f32 = overlaps.iter().copied().sum::<usize>() as f32 / overlaps.len() as f32;
        let variance: f32 = overlaps
            .iter()
            .map(|v| {
                let d = *v as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / overlaps.len() as f32;
        let std = variance.sqrt();
        if mean > 0.0 && std / mean > 0.7 {
            findings.push(Finding {
                severity: Severity::Medium,
                code: "OVERLAP_VARIANCE".into(),
                message: format!(
                    "Chunk overlap variance high (mean {:.1} tokens, cv {:.2})",
                    mean,
                    std / mean
                ),
                data: IndexMap::new(),
            });
        }
        if overlaps.iter().any(|v| *v > config.chunk_overlap * 2) {
            findings.push(Finding {
                severity: Severity::Low,
                code: "OVERLAP_EXCESS".into(),
                message: "Some chunk overlaps exceed configured overlap*2".into(),
                data: IndexMap::new(),
            });
        }
    }

    // Duplicate chunks (exact text match)
    let mut dup_map: IndexMap<&str, usize> = IndexMap::new();
    for chunk in &corpus.chunks {
        *dup_map.entry(&chunk.text).or_insert(0) += 1;
    }
    let dup_count: usize = dup_map.values().filter(|v| **v > 1).map(|v| v - 1).sum();
    if dup_count > 0 {
        findings.push(Finding {
            severity: Severity::Medium,
            code: "DUPLICATE_CHUNK".into(),
            message: format!("{dup_count} duplicate chunks detected"),
            data: IndexMap::new(),
        });
    }

    // stats
    let tokens: Vec<usize> = corpus.chunks.iter().map(|c| c.token_count).collect();
    let mut sorted = tokens.clone();
    sorted.sort_unstable();
    let p50 = percentile(&sorted, 50);
    let p95 = percentile(&sorted, 95);
    let avg = if tokens.is_empty() {
        0.0
    } else {
        tokens.iter().sum::<usize>() as f32 / tokens.len() as f32
    };
    let stats = ChunkStats {
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        avg_tokens: avg,
        min_tokens: *sorted.first().unwrap_or(&0),
        max_tokens: *sorted.last().unwrap_or(&0),
        p50_tokens: p50,
        p95_tokens: p95,
        large_chunks: oversized.len(),
        small_chunks: tiny.len(),
        duplicate_chunks: dup_count,
    };

    (stats, findings)
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
            let severity = if missing * 2 > corpus.documents.len() {
                Severity::High
            } else {
                Severity::Low
            };
            findings.push(Finding {
                severity,
                code: "MISSING_METADATA".into(),
                message: format!("{missing} docs missing {key}"),
                data: IndexMap::new(),
            });
        }
    }
    findings
}

pub fn chunk_doc_id(chunk_id: &str) -> String {
    chunk_id.split('#').next().unwrap_or(chunk_id).to_string()
}

fn count_headings(text: &str) -> usize {
    text.lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .count()
}

fn ends_mid_sentence(text: &str) -> bool {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }
    let last_non_ws = trimmed
        .chars()
        .rev()
        .find(|c| !c.is_whitespace())
        .unwrap_or('.');
    if ['.', '!', '?', '»', '”'].contains(&last_non_ws) {
        return false;
    }
    if [',', ';', ':'].contains(&last_non_ws) {
        return true;
    }
    // detect abbreviations that should not count as end (e.g., "e.g.", "Mr.")
    if is_common_abbrev(trimmed) {
        return true;
    }

    // look for sentence boundary near the end: last punctuation followed by space+capital
    let chars: Vec<char> = trimmed.chars().collect();
    let len = chars.len();
    let lookback = len.min(200);
    for idx in (len - lookback..len).rev() {
        let c = chars[idx];
        if ['.', '!', '?'].contains(&c) {
            // find next visible char after punctuation
            let mut next_non_ws = None;
            for j in idx + 1..len {
                if !chars[j].is_whitespace() {
                    next_non_ws = Some(chars[j]);
                    break;
                }
            }
            if let Some(next) = next_non_ws {
                if next.is_uppercase() {
                    return false; // likely end of sentence
                }
            }
            return true;
        }
    }

    // fallback: if trailing segment has >=6 words and no punctuation, likely mid-sentence
    let tail = trimmed
        .rsplit_once('.')
        .map(|(_, tail)| tail)
        .unwrap_or(trimmed);
    if tail.split_whitespace().count() >= 6 {
        return true;
    }
    true
}

fn is_common_abbrev(text: &str) -> bool {
    const ABBREVS: [&str; 6] = ["e.g.", "i.e.", "mr.", "mrs.", "dr.", "vs."];
    let lower = text.to_lowercase();
    ABBREVS.iter().any(|a| lower.ends_with(a))
}

fn compute_overlaps(corpus: &Corpus) -> Vec<usize> {
    let mut overlaps = Vec::new();
    let mut by_doc = IndexMap::<String, Vec<&crate::model::Chunk>>::new();
    for chunk in &corpus.chunks {
        by_doc.entry(chunk.doc_id.clone()).or_default().push(chunk);
    }
    for (_doc, chunks) in by_doc.iter_mut() {
        chunks.sort_by_key(|c| c.chunk_id.clone());
        for pair in chunks.windows(2) {
            if let [a, b] = pair {
                overlaps.push(overlap_tokens(&a.text, &b.text));
            }
        }
    }
    overlaps
}

fn overlap_tokens(a: &str, b: &str) -> usize {
    let a_tokens: Vec<&str> = a.split_whitespace().collect();
    let b_tokens: Vec<&str> = b.split_whitespace().collect();
    let max_check = a_tokens.len().min(b_tokens.len()).min(200);
    for k in (1..=max_check).rev() {
        if a_tokens.len() >= k && b_tokens.len() >= k {
            if a_tokens[a_tokens.len() - k..] == b_tokens[..k] {
                return k;
            }
        }
    }
    0
}

fn percentile(sorted: &[usize], pct: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct as f32 / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
    sorted[idx]
}
