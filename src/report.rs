use crate::cli::CompareFormat;
use crate::compare_runs::SimDiff;
use crate::config::Config;
use crate::model::{ChunkStats, ConfigSnapshot, Corpus, CoverageSummary, Finding, RetrievalResult};
use crate::retrieval::{CompareReport, ExplanationReport, QuerySpec};
use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

pub fn print_readiness(
    corpus: &Corpus,
    findings: &[Finding],
    stats: &ChunkStats,
    sim_summary: Option<&crate::model::SimSummary>,
    coverage: Option<&CoverageSummary>,
    config: &crate::config::Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let payload = JsonReport {
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        passes: None,
        fails: None,
        findings,
        retrievals: &[],
        expectations: None,
        coverage,
        chunk_stats: Some(stats.clone()),
        sim_summary: sim_summary.cloned(),
        config: Some(config_snapshot(config)),
    };
    write_artifact(artifacts, json_out, "readiness.json", &payload)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("RAG Retrieval Readiness Report");
    println!("==============================");
    println!();
    println!("Documents: {}", corpus.documents.len());
    println!("Chunks: {}", corpus.chunks.len());
    println!();
    println!(
        "Chunk config: size {} overlap {} | top_k {} | thresholds low {:.2} no-match {:.2} | embedder {:?}",
        config.chunk_size,
        config.chunk_overlap,
        config.top_k,
        config.low_sim_threshold,
        config.no_match_threshold,
        config.embedder
    );
    println!(
        "Chunk size avg {:.1} | min {} | max {} | p50 {} | p95 {}",
        stats.avg_tokens, stats.min_tokens, stats.max_tokens, stats.p50_tokens, stats.p95_tokens
    );
    print_findings(findings);
    Ok(())
}

pub fn print_simulation(
    corpus: &Corpus,
    retrievals: &[RetrievalResult],
    queries: &[QuerySpec],
    config: &Config,
    findings: &[Finding],
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let (pass, fail) = expectation_counts(retrievals, queries, config.top_k);

    let payload = JsonReport {
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        passes: Some(pass),
        fails: Some(fail),
        findings,
        retrievals,
        expectations: Some(expectation_outcomes(retrievals, queries, config.top_k)),
        coverage: None,
        chunk_stats: None,
        sim_summary: Some(crate::diagnostics::simulate_summary(
            retrievals,
            config.low_sim_threshold,
            config.no_match_threshold,
        )),
        config: Some(config_snapshot(config)),
    };
    write_artifact(artifacts, json_out, "simulation.json", &payload)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("RAG Retrieval Simulation");
    println!("========================");
    println!();
    println!("Queries: {}", retrievals.len());
    println!("Documents: {}", corpus.documents.len());
    println!("Chunks: {}", corpus.chunks.len());
    println!(
        "Config: size {} overlap {} | top_k {} | thresholds low {:.2} no-match {:.2} | embedder {:?}",
        config.chunk_size,
        config.chunk_overlap,
        config.top_k,
        config.low_sim_threshold,
        config.no_match_threshold,
        config.embedder
    );
    println!("PASS: {pass}  FAIL: {fail}");
    println!();
    if !queries.is_empty() {
        println!("Expectations (top-3 preview):");
        println!(
            "{:<12} {:<6} {:<10} {:<20} top3",
            "query", "stat", "best_rank", "expected"
        );
        for outcome in expectation_outcomes(retrievals, queries, config.top_k) {
            println!(
                "{:<12} {:<6} {:<10?} {:<20?} {:?}",
                outcome.query_id,
                outcome.status,
                outcome.best_rank,
                outcome.expected,
                outcome.top_docs
            );
        }
        println!();
    }
    if !payload
        .sim_summary
        .as_ref()
        .map(|s| !s.low_similarity_query_ids.is_empty() || !s.no_match_query_ids.is_empty())
        .unwrap_or(false)
    {
        println!("Weak/no-match queries:");
        if let Some(sum) = &payload.sim_summary {
            if !sum.low_similarity_query_ids.is_empty() {
                println!(
                    "  weak ({:.2}): {:?}",
                    config.low_sim_threshold,
                    sample_ids(&sum.low_similarity_query_ids)
                );
            }
            if !sum.no_match_query_ids.is_empty() {
                println!(
                    "  no-match ({:.2}): {:?}",
                    config.no_match_threshold,
                    sample_ids(&sum.no_match_query_ids)
                );
            }
        }
        println!();
    }
    print_findings(findings);
    Ok(())
}

pub fn print_chunks(
    corpus: &Corpus,
    findings: &[Finding],
    stats: &ChunkStats,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let payload = JsonReport {
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        passes: None,
        fails: None,
        findings,
        retrievals: &[],
        expectations: None,
        coverage: None,
        chunk_stats: Some(stats.clone()),
        sim_summary: None,
        config: None,
    };
    write_artifact(artifacts, json_out, "chunks.json", &payload)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    println!("Chunk Diagnostics");
    println!("=================");
    println!();
    let total = corpus.chunks.len() as f32;
    let avg_tokens = stats.avg_tokens;
    let large = stats.large_chunks;
    let small = stats.small_chunks;
    println!("Chunks: {}", corpus.chunks.len());
    println!("Avg tokens: {:.1}", avg_tokens);
    println!(
        "Large (>800): {} ({:.0}%) | Small (<100): {} ({:.0}%)",
        large,
        (large as f32 / total.max(1.0)) * 100.0,
        small,
        (small as f32 / total.max(1.0)) * 100.0
    );
    println!();
    print_findings(findings);
    Ok(())
}

pub fn print_coverage(
    corpus: &Corpus,
    findings: &[Finding],
    coverage: Option<&CoverageSummary>,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let payload = JsonReport {
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        passes: coverage.map(|c| c.good),
        fails: coverage.map(|c| c.weak + c.none),
        findings,
        retrievals: &[],
        expectations: None,
        coverage,
        chunk_stats: None,
        sim_summary: None,
        config: Some(config_snapshot(config)),
    };
    write_artifact(artifacts, json_out, "coverage.json", &payload)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    if let Some(cov) = coverage {
        println!("Retrieval Coverage");
        println!("==================");
        println!();
        println!("Queries: {}", cov.queries);
        println!("Good: {}  Weak: {}  None: {}", cov.good, cov.weak, cov.none);
        println!(
            "Config: size {} overlap {} | top_k {} | thresholds low {:.2} no-match {:.2} | embedder {:?}",
            config.chunk_size,
            config.chunk_overlap,
            config.top_k,
            config.low_sim_threshold,
            config.no_match_threshold,
            config.embedder
        );
    } else {
        println!("Topic Coverage");
        println!("==============");
        println!();
    }
    print_findings(findings);
    Ok(())
}

pub fn print_explanation(
    query: &str,
    report: &ExplanationReport,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
) -> Result<()> {
    write_artifact(artifacts, json_out, "explain.json", report)?;
    println!("Retrieval Explanation");
    println!("=====================");
    println!();
    println!("Query: {query}");
    println!();
    for item in &report.ranked {
        println!("{} {} (doc: {})", item.rank, item.chunk_id, item.doc_id);
        println!("  score: {:.3}", item.explanation.total_score);
        println!("  components:");
        println!("    semantic: {:.3}", item.explanation.similarity);
        println!(
            "    keyword overlap: {} ({:.2})",
            item.explanation.keyword_overlap, item.explanation.keyword_overlap_norm
        );
        println!(
            "    metadata boost: {:.3} | length penalty: {:.3}",
            item.explanation.metadata_boost, item.explanation.length_penalty
        );
        println!(
            "    phrase match: {} | tokens: {}",
            item.explanation.phrase_match, item.explanation.token_count
        );
    }
    Ok(())
}

pub fn print_comparison(
    query: &str,
    report: &CompareReport,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
) -> Result<()> {
    write_artifact(artifacts, json_out, "compare.json", report)?;
    println!("Retrieval Compare");
    println!("=================");
    println!();
    println!("Query: {query}");
    println!();
    for item in &report.ranked {
        println!("{} {} (doc: {})", item.rank, item.chunk_id, item.doc_id);
        println!("  similarity: {:.3}", item.score);
        println!(
            "  keyword overlap: {} | phrase match: {} | tokens: {}",
            item.explanation.keyword_overlap,
            item.explanation.phrase_match,
            item.explanation.token_count
        );
    }
    Ok(())
}

fn print_findings(findings: &[Finding]) {
    if findings.is_empty() {
        println!("No findings.");
        return;
    }
    for finding in findings {
        println!(
            "- {}: {}",
            severity_label(&finding.severity),
            finding.message
        );
    }
}

#[derive(Serialize)]
struct JsonReport<'a> {
    documents: usize,
    chunks: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    passes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fails: Option<usize>,
    findings: &'a [Finding],
    #[serde(skip_serializing_if = "slice_is_empty")]
    retrievals: &'a [RetrievalResult],
    #[serde(skip_serializing_if = "Option::is_none")]
    expectations: Option<Vec<ExpectationOutcome>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<&'a CoverageSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_stats: Option<ChunkStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sim_summary: Option<crate::model::SimSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<ConfigSnapshot>,
}

fn slice_is_empty<T>(slice: &[T]) -> bool {
    slice.is_empty()
}

fn severity_label(sev: &crate::model::Severity) -> &'static str {
    match sev {
        crate::model::Severity::High => "HIGH",
        crate::model::Severity::Medium => "MEDIUM",
        crate::model::Severity::Low => "LOW",
        crate::model::Severity::Info => "INFO",
        crate::model::Severity::Fail => "FAIL",
    }
}

fn write_artifact<T: Serialize>(
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    name: &str,
    payload: &T,
) -> Result<()> {
    if let Some(path) = json_out {
        let data = serde_json::to_vec_pretty(payload)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data)?;
        return Ok(());
    }
    if let Some(dir) = artifacts {
        let path = dir.join(name);
        let data = serde_json::to_vec_pretty(payload)?;
        fs::create_dir_all(dir)?;
        fs::write(path, data)?;
    }
    Ok(())
}

fn config_snapshot(cfg: &Config) -> ConfigSnapshot {
    ConfigSnapshot {
        chunk_size: cfg.chunk_size,
        chunk_overlap: cfg.chunk_overlap,
        min_tokens: cfg.min_tokens,
        max_tokens: cfg.max_tokens,
        top_k: cfg.top_k,
        dominant_threshold: cfg.dominant_threshold,
        low_sim_threshold: cfg.low_sim_threshold,
        no_match_threshold: cfg.no_match_threshold,
        embedder: format!("{:?}", cfg.embedder),
        seed: cfg.seed,
    }
}

fn sample_ids(ids: &[String]) -> Vec<String> {
    ids.iter().take(3).cloned().collect()
}

pub fn print_run_comparison(
    diff: &SimDiff,
    format: CompareFormat,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    write_artifact(artifacts, json_out, "compare_runs.json", diff)?;
    if json {
        println!("{}", serde_json::to_string_pretty(diff)?);
        return Ok(());
    }
    match format {
        CompareFormat::Summary => {
            println!("Retrieval comparison");
            println!("====================");
            println!();
            println!(
                "Queries: before {} | after {}",
                diff.queries_before, diff.queries_after
            );
            println!(
                "Avg top-1 similarity: before {:.3} | after {:.3}",
                diff.avg_top1_similarity_before, diff.avg_top1_similarity_after
            );
            println!(
                "Weak matches: before {} | after {}",
                diff.weak_before, diff.weak_after
            );
            println!(
                "No matches: before {} | after {}",
                diff.no_match_before, diff.no_match_after
            );
            println!("\nTop-1 documents (counts):");
            for d in &diff.top1_docs {
                println!(
                    "- {}: {} -> {} (Δ {})",
                    d.doc_id, d.before, d.after, d.delta
                );
            }
        }
        CompareFormat::Table => {
            println!("Metric                     Before   After   Delta");
            println!("-------------------------------------------------");
            println!(
                "Avg top-1 similarity       {:.3}   {:.3}   {:+.3}",
                diff.avg_top1_similarity_before,
                diff.avg_top1_similarity_after,
                diff.avg_top1_similarity_after - diff.avg_top1_similarity_before
            );
            println!(
                "Weak matches               {:>6}   {:>5}   {:+}",
                diff.weak_before,
                diff.weak_after,
                diff.weak_after as isize - diff.weak_before as isize
            );
            println!(
                "No matches                 {:>6}   {:>5}   {:+}",
                diff.no_match_before,
                diff.no_match_after,
                diff.no_match_after as isize - diff.no_match_before as isize
            );
            println!("\nTop-1 document deltas:");
            for d in &diff.top1_docs {
                println!(
                    "{:<24} {:>6} -> {:<6} ({:+})",
                    d.doc_id, d.before, d.after, d.delta
                );
            }
        }
    }
    Ok(())
}

fn expectation_counts(
    retrievals: &[RetrievalResult],
    queries: &[QuerySpec],
    top_k: usize,
) -> (usize, usize) {
    let mut pass = 0;
    let mut fail = 0;
    for spec in queries {
        if spec.expect_docs.is_empty() {
            continue;
        }
        let result = retrievals.iter().find(|r| r.query_id == spec.id);
        if let Some(r) = result {
            let mut ok = false;
            for doc in &spec.expect_docs {
                if r.ranked.iter().any(|c| {
                    crate::diagnostics::chunk_doc_id(&c.chunk_id) == *doc && c.rank <= top_k
                }) {
                    ok = true;
                    break;
                }
            }
            if ok {
                pass += 1;
            } else {
                fail += 1;
            }
        } else {
            fail += 1;
        }
    }
    (pass, fail)
}

#[derive(Serialize, Clone)]
struct ExpectationOutcome {
    query_id: String,
    expected: Vec<String>,
    best_rank: Option<usize>,
    top_docs: Vec<String>,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_doc_ranks: Option<Vec<RankEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_ranks: Option<Vec<RankEntry>>,
}

#[derive(Serialize, Clone)]
struct RankEntry {
    doc: String,
    rank: usize,
}

fn expectation_outcomes(
    retrievals: &[RetrievalResult],
    queries: &[QuerySpec],
    top_k: usize,
) -> Vec<ExpectationOutcome> {
    let mut out = Vec::new();
    for spec in queries {
        if spec.expect_docs.is_empty() {
            continue;
        }
        let result = retrievals.iter().find(|r| r.query_id == spec.id);
        let (best_rank, top_docs) = if let Some(r) = result {
            let best = spec
                .expect_docs
                .iter()
                .filter_map(|doc| {
                    r.ranked
                        .iter()
                        .find(|c| crate::diagnostics::chunk_doc_id(&c.chunk_id) == *doc)
                        .map(|c| c.rank)
                })
                .min();
            let top_docs: Vec<String> = r
                .ranked
                .iter()
                .take(3)
                .map(|c| crate::diagnostics::chunk_doc_id(&c.chunk_id))
                .collect();
            (best, top_docs)
        } else {
            (None, Vec::new())
        };

        let top_doc_ranks = result.map(|r| {
            r.ranked
                .iter()
                .take(3)
                .map(|c| RankEntry {
                    doc: crate::diagnostics::chunk_doc_id(&c.chunk_id),
                    rank: c.rank,
                })
                .collect::<Vec<_>>()
        });

        let expected_ranks = result.map(|r| {
            spec.expect_docs
                .iter()
                .filter_map(|doc| {
                    r.ranked
                        .iter()
                        .find(|c| crate::diagnostics::chunk_doc_id(&c.chunk_id) == *doc)
                        .map(|c| RankEntry {
                            doc: doc.clone(),
                            rank: c.rank,
                        })
                })
                .collect::<Vec<_>>()
        });

        let status = if let Some(rank) = best_rank {
            if rank <= top_k {
                "PASS"
            } else {
                "FAIL"
            }
        } else {
            "FAIL"
        };
        out.push(ExpectationOutcome {
            query_id: spec.id.clone(),
            expected: spec.expect_docs.clone(),
            best_rank,
            top_docs,
            status: status.to_string(),
            top_doc_ranks,
            expected_ranks,
        });
    }
    out
}
