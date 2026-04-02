use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
struct DocFreq {
    doc_id: String,
    count: usize,
}

#[derive(Debug, Deserialize)]
struct SimSummary {
    queries: usize,
    avg_top1_similarity: f32,
    low_similarity_queries: usize,
    no_match_queries: usize,
    #[serde(default)]
    top1_freq: Vec<DocFreq>,
    #[allow(dead_code)]
    #[serde(default)]
    low_similarity_query_ids: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    no_match_query_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RankedChunk {
    chunk_id: String,
    score: f32,
}

#[derive(Debug, Deserialize)]
struct RetrievalResult {
    ranked: Vec<RankedChunk>,
}

#[derive(Debug, Deserialize)]
struct JsonReport {
    sim_summary: Option<SimSummary>,
    #[serde(default)]
    retrievals: Vec<RetrievalResult>,
    #[serde(default)]
    config: Option<ReportConfig>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ReportConfig {
    #[serde(default = "default_low_sim_threshold")]
    low_sim_threshold: f32,
    #[serde(default = "default_no_match_threshold")]
    no_match_threshold: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiffDocFreq {
    pub doc_id: String,
    pub before: usize,
    pub after: usize,
    pub delta: isize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SimDiff {
    pub queries_before: usize,
    pub queries_after: usize,
    pub query_count_mismatch: bool,
    pub avg_top1_similarity_before: f32,
    pub avg_top1_similarity_after: f32,
    pub weak_before: usize,
    pub weak_after: usize,
    pub no_match_before: usize,
    pub no_match_after: usize,
    pub top1_dominant_doc_before: Option<String>,
    pub top1_dominant_rate_before: f32,
    pub top1_dominant_doc_after: Option<String>,
    pub top1_dominant_rate_after: f32,
    pub top1_docs: Vec<DiffDocFreq>,
    pub verdict: String,
    pub improved_metrics: Vec<String>,
    pub regressed_metrics: Vec<String>,
}

pub fn compare_runs(baseline: &Path, improved: &Path) -> Result<SimDiff> {
    let before = load_report(baseline)?;
    let after = load_report(improved)?;

    let before_sum = summarize(&before);
    let after_sum = summarize(&after);

    let top1_docs = merge_top1(&before_sum.top1_freq, &after_sum.top1_freq);
    let (top1_dominant_doc_before, top1_dominant_rate_before) =
        dominant_top1(&before_sum.top1_freq, before_sum.queries);
    let (top1_dominant_doc_after, top1_dominant_rate_after) =
        dominant_top1(&after_sum.top1_freq, after_sum.queries);

    let (verdict, improved_metrics, regressed_metrics) = classify(
        QualitySignals {
            avg_top1_similarity: before_sum.avg_top1_similarity,
            weak_matches: before_sum.low_similarity_queries,
            no_matches: before_sum.no_match_queries,
            top1_dominant_rate: top1_dominant_rate_before,
        },
        QualitySignals {
            avg_top1_similarity: after_sum.avg_top1_similarity,
            weak_matches: after_sum.low_similarity_queries,
            no_matches: after_sum.no_match_queries,
            top1_dominant_rate: top1_dominant_rate_after,
        },
    );

    Ok(SimDiff {
        queries_before: before_sum.queries,
        queries_after: after_sum.queries,
        query_count_mismatch: before_sum.queries != after_sum.queries,
        avg_top1_similarity_before: before_sum.avg_top1_similarity,
        avg_top1_similarity_after: after_sum.avg_top1_similarity,
        weak_before: before_sum.low_similarity_queries,
        weak_after: after_sum.low_similarity_queries,
        no_match_before: before_sum.no_match_queries,
        no_match_after: after_sum.no_match_queries,
        top1_dominant_doc_before,
        top1_dominant_rate_before,
        top1_dominant_doc_after,
        top1_dominant_rate_after,
        top1_docs,
        verdict,
        improved_metrics,
        regressed_metrics,
    })
}

#[derive(Default)]
struct SimpleSum {
    queries: usize,
    avg_top1_similarity: f32,
    low_similarity_queries: usize,
    no_match_queries: usize,
    top1_freq: Vec<DocFreq>,
}

fn load_report(path: &Path) -> Result<JsonReport> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let rep: JsonReport =
        serde_json::from_reader(f).with_context(|| format!("parsing {}", path.display()))?;
    Ok(rep)
}

fn summarize(rep: &JsonReport) -> SimpleSum {
    if let Some(sim) = &rep.sim_summary {
        return SimpleSum {
            queries: sim.queries,
            avg_top1_similarity: sim.avg_top1_similarity,
            low_similarity_queries: sim.low_similarity_queries,
            no_match_queries: sim.no_match_queries,
            top1_freq: sim.top1_freq.clone(),
        };
    }
    // Fallback: derive from retrievals if sim_summary missing
    let queries = rep.retrievals.len();
    let cfg = rep.config.unwrap_or(ReportConfig {
        low_sim_threshold: default_low_sim_threshold(),
        no_match_threshold: default_no_match_threshold(),
    });
    let mut sum_sim = 0f32;
    let mut low_similarity_queries = 0usize;
    let mut no_match_queries = 0usize;
    let mut top1_freq: HashMap<String, usize> = HashMap::new();
    for r in &rep.retrievals {
        let top_score = r.ranked.first().map(|top| top.score).unwrap_or(0.0);
        sum_sim += top_score;
        if top_score < cfg.low_sim_threshold {
            low_similarity_queries += 1;
        }
        if top_score < cfg.no_match_threshold {
            no_match_queries += 1;
        }
        if let Some(top) = r.ranked.first() {
            let doc = top.chunk_id.split('#').next().unwrap_or(&top.chunk_id);
            *top1_freq.entry(doc.to_string()).or_insert(0) += 1;
        }
    }
    let avg = if queries > 0 {
        sum_sim / queries as f32
    } else {
        0.0
    };
    let freqs = top1_freq
        .into_iter()
        .map(|(doc_id, count)| DocFreq { doc_id, count })
        .collect::<Vec<_>>();
    let mut freqs = freqs;
    freqs.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    SimpleSum {
        queries,
        avg_top1_similarity: avg,
        low_similarity_queries,
        no_match_queries,
        top1_freq: freqs,
    }
}

fn merge_top1(before: &[DocFreq], after: &[DocFreq]) -> Vec<DiffDocFreq> {
    let mut map_before: HashMap<String, usize> = HashMap::new();
    for d in before {
        map_before.insert(d.doc_id.clone(), d.count);
    }
    let mut map_after: HashMap<String, usize> = HashMap::new();
    for d in after {
        map_after.insert(d.doc_id.clone(), d.count);
    }
    let mut docs: Vec<String> = map_before.keys().chain(map_after.keys()).cloned().collect();
    docs.sort();
    docs.dedup();
    let mut diffs = Vec::new();
    for doc in docs {
        let b = *map_before.get(&doc).unwrap_or(&0);
        let a = *map_after.get(&doc).unwrap_or(&0);
        diffs.push(DiffDocFreq {
            doc_id: doc,
            before: b,
            after: a,
            delta: a as isize - b as isize,
        });
    }
    diffs.sort_by(|x, y| y.after.cmp(&x.after)); // sort by after freq
    diffs
}

fn dominant_top1(freqs: &[DocFreq], queries: usize) -> (Option<String>, f32) {
    if queries == 0 {
        return (None, 0.0);
    }
    let mut best_doc: Option<String> = None;
    let mut best_count = 0usize;
    for d in freqs {
        let take = if d.count > best_count {
            true
        } else if d.count == best_count {
            match &best_doc {
                Some(cur) => d.doc_id < *cur,
                None => true,
            }
        } else {
            false
        };
        if take {
            best_count = d.count;
            best_doc = Some(d.doc_id.clone());
        }
    }
    (best_doc, best_count as f32 / queries as f32)
}

#[derive(Clone, Copy)]
struct QualitySignals {
    avg_top1_similarity: f32,
    weak_matches: usize,
    no_matches: usize,
    top1_dominant_rate: f32,
}

fn classify(before: QualitySignals, after: QualitySignals) -> (String, Vec<String>, Vec<String>) {
    let mut improved = Vec::new();
    let mut regressed = Vec::new();
    let eps = 1e-6f32;

    if after.avg_top1_similarity > before.avg_top1_similarity + eps {
        improved.push("avg_top1_similarity".to_string());
    } else if after.avg_top1_similarity + eps < before.avg_top1_similarity {
        regressed.push("avg_top1_similarity".to_string());
    }
    if after.weak_matches < before.weak_matches {
        improved.push("weak_matches".to_string());
    } else if after.weak_matches > before.weak_matches {
        regressed.push("weak_matches".to_string());
    }
    if after.no_matches < before.no_matches {
        improved.push("no_matches".to_string());
    } else if after.no_matches > before.no_matches {
        regressed.push("no_matches".to_string());
    }
    if after.top1_dominant_rate + eps < before.top1_dominant_rate {
        improved.push("top1_dominant_rate".to_string());
    } else if after.top1_dominant_rate > before.top1_dominant_rate + eps {
        regressed.push("top1_dominant_rate".to_string());
    }

    let verdict = if improved.len() > regressed.len() {
        "IMPROVED"
    } else if regressed.len() > improved.len() {
        "REGRESSED"
    } else {
        "NEUTRAL"
    };

    (verdict.to_string(), improved, regressed)
}

fn default_low_sim_threshold() -> f32 {
    0.35
}

fn default_no_match_threshold() -> f32 {
    0.25
}

#[cfg(test)]
mod tests {
    use super::{classify, dominant_top1, QualitySignals};
    use super::{summarize, JsonReport, RankedChunk, ReportConfig, RetrievalResult};

    #[test]
    fn classify_improved_case() {
        let (verdict, improved, regressed) = classify(
            QualitySignals {
                avg_top1_similarity: 0.50,
                weak_matches: 10,
                no_matches: 5,
                top1_dominant_rate: 0.60,
            },
            QualitySignals {
                avg_top1_similarity: 0.70,
                weak_matches: 3,
                no_matches: 1,
                top1_dominant_rate: 0.30,
            },
        );
        assert_eq!(verdict, "IMPROVED");
        assert!(!improved.is_empty());
        assert!(regressed.is_empty());
    }

    #[test]
    fn classify_regressed_case() {
        let (verdict, improved, regressed) = classify(
            QualitySignals {
                avg_top1_similarity: 0.70,
                weak_matches: 2,
                no_matches: 1,
                top1_dominant_rate: 0.30,
            },
            QualitySignals {
                avg_top1_similarity: 0.60,
                weak_matches: 8,
                no_matches: 4,
                top1_dominant_rate: 0.80,
            },
        );
        assert_eq!(verdict, "REGRESSED");
        assert!(improved.is_empty());
        assert!(!regressed.is_empty());
    }

    #[test]
    fn summarize_fallback_computes_low_and_no_match() {
        let rep = JsonReport {
            sim_summary: None,
            retrievals: vec![
                RetrievalResult {
                    ranked: vec![RankedChunk {
                        chunk_id: "a.md#0".into(),
                        score: 0.5,
                    }],
                },
                RetrievalResult { ranked: vec![] },
            ],
            config: Some(ReportConfig {
                low_sim_threshold: 0.4,
                no_match_threshold: 0.3,
            }),
        };
        let s = summarize(&rep);
        assert_eq!(s.queries, 2);
        assert_eq!(s.low_similarity_queries, 1);
        assert_eq!(s.no_match_queries, 1);
    }

    #[test]
    fn dominant_top1_tie_breaks_by_doc_id() {
        let freqs = vec![
            super::DocFreq {
                doc_id: "z.md".into(),
                count: 3,
            },
            super::DocFreq {
                doc_id: "a.md".into(),
                count: 3,
            },
        ];
        let (doc, rate) = dominant_top1(&freqs, 10);
        assert_eq!(doc.as_deref(), Some("a.md"));
        assert!((rate - 0.3).abs() < 1e-6);
    }
}
