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
}

#[derive(Debug, serde::Serialize)]
pub struct DiffDocFreq {
    pub doc_id: String,
    pub before: usize,
    pub after: usize,
    pub delta: isize,
}

#[derive(Debug, serde::Serialize)]
pub struct SimDiff {
    pub queries_before: usize,
    pub queries_after: usize,
    pub avg_top1_similarity_before: f32,
    pub avg_top1_similarity_after: f32,
    pub weak_before: usize,
    pub weak_after: usize,
    pub no_match_before: usize,
    pub no_match_after: usize,
    pub top1_docs: Vec<DiffDocFreq>,
}

pub fn compare_runs(baseline: &Path, improved: &Path) -> Result<SimDiff> {
    let before = load_report(baseline)?;
    let after = load_report(improved)?;

    let before_sum = summarize(&before);
    let after_sum = summarize(&after);

    let top1_docs = merge_top1(&before_sum.top1_freq, &after_sum.top1_freq);

    Ok(SimDiff {
        queries_before: before_sum.queries,
        queries_after: after_sum.queries,
        avg_top1_similarity_before: before_sum.avg_top1_similarity,
        avg_top1_similarity_after: after_sum.avg_top1_similarity,
        weak_before: before_sum.low_similarity_queries,
        weak_after: after_sum.low_similarity_queries,
        no_match_before: before_sum.no_match_queries,
        no_match_after: after_sum.no_match_queries,
        top1_docs,
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
    let mut sum_sim = 0f32;
    let mut top1_freq: HashMap<String, usize> = HashMap::new();
    for r in &rep.retrievals {
        if let Some(top) = r.ranked.first() {
            sum_sim += top.score;
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
        .collect();
    SimpleSum {
        queries,
        avg_top1_similarity: avg,
        low_similarity_queries: 0,
        no_match_queries: 0,
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
