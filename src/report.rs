use crate::cli::CompareFormat;
use crate::compare_runs::SimDiff;
use crate::config::Config;
use crate::model::{
    ChunkStats, ConfigSnapshot, Corpus, CoverageSummary, Finding, FixReport, OptimizeSummary,
    RetrievalResult,
};
use crate::retrieval::{CompareReport, ExplanationReport, QuerySpec};
use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub struct OutputOpts<'a> {
    pub artifacts: Option<&'a PathBuf>,
    pub json_out: Option<&'a PathBuf>,
    pub html_out: Option<&'a PathBuf>,
    pub json: bool,
}

pub fn print_readiness(
    corpus: &Corpus,
    findings: &[Finding],
    stats: &ChunkStats,
    sim_summary: Option<&crate::model::SimSummary>,
    coverage: Option<&CoverageSummary>,
    config: &crate::config::Config,
    output: OutputOpts<'_>,
) -> Result<()> {
    let payload = JsonReport {
        meta: report_meta("readiness"),
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
    write_artifact(output, "readiness.json", &payload)?;
    if output.json {
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
    output: OutputOpts<'_>,
) -> Result<()> {
    let expectation_rows = expectation_outcomes(retrievals, queries, config.top_k);
    let pass = expectation_rows
        .iter()
        .filter(|r| r.status == "PASS")
        .count();
    let fail = expectation_rows
        .iter()
        .filter(|r| r.status == "FAIL")
        .count();

    let payload = JsonReport {
        meta: report_meta("simulate"),
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        passes: Some(pass),
        fails: Some(fail),
        findings,
        retrievals,
        expectations: Some(expectation_rows.clone()),
        coverage: None,
        chunk_stats: None,
        sim_summary: Some(crate::diagnostics::simulate_summary(
            retrievals,
            config.low_sim_threshold,
            config.no_match_threshold,
        )),
        config: Some(config_snapshot(config)),
    };
    write_artifact(output, "simulation.json", &payload)?;
    if output.json {
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
    if !expectation_rows.is_empty() {
        println!("Expectations (top-3 preview):");
        println!(
            "{:<12} {:<6} {:<10} {:<20} top3",
            "query", "stat", "best_rank", "expected"
        );
        for outcome in expectation_rows {
            let best_rank = outcome
                .best_rank
                .map(|r| r.to_string())
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:<12} {:<6} {:<10} {:<20?} {:?}",
                outcome.query_id, outcome.status, best_rank, outcome.expected, outcome.top_docs
            );
        }
        println!();
    }
    if payload
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
    config: &Config,
    output: OutputOpts<'_>,
) -> Result<()> {
    let payload = JsonReport {
        meta: report_meta("chunks"),
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
    write_artifact(output, "chunks.json", &payload)?;
    if output.json {
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
        "Large (>{}): {} ({:.0}%) | Small (<{}): {} ({:.0}%)",
        config.max_tokens,
        large,
        (large as f32 / total.max(1.0)) * 100.0,
        config.min_tokens,
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
    output: OutputOpts<'_>,
) -> Result<()> {
    let payload = JsonReport {
        meta: report_meta("coverage"),
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
    write_artifact(output, "coverage.json", &payload)?;
    if output.json {
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
    output: OutputOpts<'_>,
) -> Result<()> {
    let payload = ExplainJson {
        meta: report_meta("explain"),
        query,
        report,
    };
    write_artifact(output, "explain.json", &payload)?;
    let html = render_explain_html(query, report);
    write_html_artifact(output, "explain.html", &html)?;
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
            "    keyword overlap: {} ({:.2}) | keyword boost: {:.3}",
            item.explanation.keyword_overlap,
            item.explanation.keyword_overlap_norm,
            item.explanation.keyword_boost
        );
        println!(
            "    metadata boost: {:.3} | length penalty: {:.3}",
            item.explanation.metadata_boost, item.explanation.length_penalty
        );
        println!(
            "    phrase match: {} | phrase boost: {:.3} | tokens: {}",
            item.explanation.phrase_match,
            item.explanation.phrase_boost,
            item.explanation.token_count
        );
    }
    Ok(())
}

pub fn print_comparison(query: &str, report: &CompareReport, output: OutputOpts<'_>) -> Result<()> {
    let payload = CompareQueryJson {
        meta: report_meta("compare-query"),
        query,
        report,
    };
    write_artifact(output, "compare.json", &payload)?;
    let html = render_compare_html(query, report);
    write_html_artifact(output, "compare.html", &html)?;
    println!("Retrieval Compare");
    println!("=================");
    println!();
    println!("Query: {query}");
    println!();
    for item in &report.ranked {
        println!("{} {} (doc: {})", item.rank, item.chunk_id, item.doc_id);
        println!("  score: {:.3}", item.score);
        println!(
            "  semantic: {:.3} | keyword overlap: {} | phrase match: {} | tokens: {}",
            item.explanation.similarity,
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
    meta: ReportMeta,
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

#[derive(Serialize)]
struct ReportMeta {
    schema_version: &'static str,
    tool_version: &'static str,
    command: &'static str,
}

#[derive(Serialize)]
struct ExplainJson<'a> {
    meta: ReportMeta,
    query: &'a str,
    report: &'a ExplanationReport,
}

#[derive(Serialize)]
struct CompareQueryJson<'a> {
    meta: ReportMeta,
    query: &'a str,
    report: &'a CompareReport,
}

#[derive(Serialize)]
struct CompareRunsJson {
    meta: ReportMeta,
    #[serde(flatten)]
    diff: SimDiff,
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

fn write_artifact<T: Serialize>(output: OutputOpts<'_>, name: &str, payload: &T) -> Result<()> {
    if let Some(path) = output.json_out {
        let data = serde_json::to_vec_pretty(payload)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data)?;
        return Ok(());
    }
    if let Some(dir) = output.artifacts {
        let path = dir.join(name);
        let data = serde_json::to_vec_pretty(payload)?;
        fs::create_dir_all(dir)?;
        fs::write(path, data)?;
    }
    Ok(())
}

fn write_html_artifact(output: OutputOpts<'_>, name: &str, html: &str) -> Result<()> {
    if let Some(path) = output.html_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, html)?;
        return Ok(());
    }
    if let Some(dir) = output.artifacts {
        let path = dir.join(name);
        fs::create_dir_all(dir)?;
        fs::write(path, html)?;
    }
    Ok(())
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn render_explain_html(query: &str, report: &ExplanationReport) -> String {
    let mut rows = String::new();
    for item in &report.ranked {
        rows.push_str(&format!(
            "<tr>\
                <td>{}</td>\
                <td><code>{}</code></td>\
                <td><code>{}</code></td>\
                <td>{:.3}</td>\
                <td>{:.3}</td>\
                <td>{} ({:.2})</td>\
                <td>{}</td>\
                <td>{}</td>\
            </tr>",
            item.rank,
            html_escape(&item.chunk_id),
            html_escape(&item.doc_id),
            item.explanation.total_score,
            item.explanation.similarity,
            item.explanation.keyword_overlap,
            item.explanation.keyword_overlap_norm,
            if item.explanation.phrase_match {
                "yes"
            } else {
                "no"
            },
            item.explanation.token_count
        ));
    }
    format!(
        "<!doctype html>\
<html lang=\"en\">\
<head>\
  <meta charset=\"utf-8\"/>\
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\
  <title>RAGLens Explain</title>\
  <style>\
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}\
    h1 {{ margin: 0 0 8px; }}\
    .query {{ margin: 0 0 18px; color: #444; }}\
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}\
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}\
    th {{ background: #f7f7f7; }}\
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }}\
  </style>\
</head>\
<body>\
  <h1>RAGLens Retrieval Explanation</h1>\
  <p class=\"query\"><strong>Query:</strong> {}</p>\
  <table>\
    <thead>\
      <tr>\
        <th>Rank</th><th>Chunk</th><th>Document</th><th>Total score</th>\
        <th>Semantic</th><th>Keyword overlap</th><th>Phrase match</th><th>Tokens</th>\
      </tr>\
    </thead>\
    <tbody>{}</tbody>\
  </table>\
</body>\
</html>",
        html_escape(query),
        rows
    )
}

fn render_compare_html(query: &str, report: &CompareReport) -> String {
    let mut rows = String::new();
    for item in &report.ranked {
        rows.push_str(&format!(
            "<tr>\
                <td>{}</td>\
                <td><code>{}</code></td>\
                <td><code>{}</code></td>\
                <td>{:.3}</td>\
                <td>{:.3}</td>\
                <td>{}</td>\
            </tr>",
            item.rank,
            html_escape(&item.chunk_id),
            html_escape(&item.doc_id),
            item.score,
            item.explanation.similarity,
            item.explanation.keyword_overlap
        ));
    }
    format!(
        "<!doctype html>\
<html lang=\"en\">\
<head>\
  <meta charset=\"utf-8\"/>\
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\
  <title>RAGLens Compare</title>\
  <style>\
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}\
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}\
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}\
    th {{ background: #f7f7f7; }}\
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }}\
  </style>\
</head>\
<body>\
  <h1>RAGLens Retrieval Compare</h1>\
  <p><strong>Query:</strong> {}</p>\
  <table>\
    <thead>\
      <tr><th>Rank</th><th>Chunk</th><th>Document</th><th>Score</th><th>Semantic</th><th>Keyword overlap</th></tr>\
    </thead>\
    <tbody>{}</tbody>\
  </table>\
</body>\
</html>",
        html_escape(query),
        rows
    )
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

fn report_meta(command: &'static str) -> ReportMeta {
    ReportMeta {
        schema_version: "1",
        tool_version: env!("CARGO_PKG_VERSION"),
        command,
    }
}

pub fn print_run_comparison(
    diff: &SimDiff,
    format: CompareFormat,
    output: OutputOpts<'_>,
) -> Result<()> {
    let payload = CompareRunsJson {
        meta: report_meta("compare-runs"),
        diff: diff.clone(),
    };
    write_artifact(output, "compare_runs.json", &payload)?;
    if output.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    match format {
        CompareFormat::Summary => {
            println!("Retrieval comparison");
            println!("====================");
            println!();
            println!("Verdict: {}", diff.verdict);
            println!(
                "Queries: before {} | after {}",
                diff.queries_before, diff.queries_after
            );
            if diff.query_count_mismatch {
                println!(
                    "WARNING: query count mismatch; comparisons may be misleading (before {} vs after {})",
                    diff.queries_before, diff.queries_after
                );
            }
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
            println!(
                "Top-1 dominant: before {} ({:.0}%) | after {} ({:.0}%)",
                diff.top1_dominant_doc_before.as_deref().unwrap_or("<none>"),
                diff.top1_dominant_rate_before * 100.0,
                diff.top1_dominant_doc_after.as_deref().unwrap_or("<none>"),
                diff.top1_dominant_rate_after * 100.0
            );
            if !diff.improved_metrics.is_empty() {
                println!("Improved metrics: {}", diff.improved_metrics.join(", "));
            }
            if !diff.regressed_metrics.is_empty() {
                println!("Regressed metrics: {}", diff.regressed_metrics.join(", "));
            }
            println!("\nTop-1 documents (counts):");
            for d in &diff.top1_docs {
                println!(
                    "- {}: {} -> {} (Δ {})",
                    d.doc_id, d.before, d.after, d.delta
                );
            }
        }
        CompareFormat::Table => {
            println!("Verdict: {}", diff.verdict);
            if diff.query_count_mismatch {
                println!(
                    "WARNING: query count mismatch (before {} vs after {})",
                    diff.queries_before, diff.queries_after
                );
            }
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
            println!(
                "Top-1 dominant rate        {:.3}   {:.3}   {:+.3}",
                diff.top1_dominant_rate_before,
                diff.top1_dominant_rate_after,
                diff.top1_dominant_rate_after - diff.top1_dominant_rate_before
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

pub fn print_optimize(summary: &OptimizeSummary, output: OutputOpts<'_>) -> Result<()> {
    #[derive(Serialize)]
    struct OptimizeJson<'a> {
        meta: ReportMeta,
        optimize: &'a OptimizeSummary,
    }

    let payload = OptimizeJson {
        meta: report_meta("optimize"),
        optimize: summary,
    };
    write_artifact(output, "optimize.json", &payload)?;
    if output.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("Chunking Optimization");
    println!("=====================");
    println!();
    println!("Queries file: {}", summary.queries_file);
    println!(
        "Candidates: considered {} | skipped {}",
        summary.considered, summary.skipped
    );
    if let Some(best) = &summary.best {
        println!(
            "Best: chunk_size={} overlap={} score={:.2}",
            best.chunk_size, best.chunk_overlap, best.score
        );
        println!(
            "  avg_top1_similarity {:.3} | weak {} | no-match {} | expectation fails {} | dominant {:.0}%",
            best.avg_top1_similarity,
            best.low_similarity_queries,
            best.no_match_queries,
            best.expectation_failures,
            best.dominant_rate * 100.0
        );
    } else {
        println!("No valid candidates.");
        return Ok(());
    }

    println!();
    println!(
        "Top {} candidates:",
        summary.top_n.min(summary.candidates.len())
    );
    println!("size  overlap  score   sim    weak  none  exp_fail  dom%");
    for c in summary.candidates.iter().take(summary.top_n) {
        println!(
            "{:<5} {:<7} {:<6.2} {:<6.3} {:<5} {:<5} {:<9} {:<5.0}",
            c.chunk_size,
            c.chunk_overlap,
            c.score,
            c.avg_top1_similarity,
            c.low_similarity_queries,
            c.no_match_queries,
            c.expectation_failures,
            c.dominant_rate * 100.0
        );
    }
    Ok(())
}

pub fn print_fix(fix: &FixReport, output: OutputOpts<'_>) -> Result<()> {
    #[derive(Serialize)]
    struct FixJson<'a> {
        meta: ReportMeta,
        fix: &'a FixReport,
    }

    let payload = FixJson {
        meta: report_meta("fix"),
        fix,
    };
    write_artifact(output, "fix.json", &payload)?;
    if output.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("Retrieval Fix Advisor");
    println!("=====================");
    println!();
    println!("Issue: {}", fix.issue);
    println!();
    println!("Likely causes:");
    for cause in &fix.likely_causes {
        println!("- {}", cause);
    }
    println!();
    println!("Try first: {}", fix.first_fix);
    println!("Then rerun: {}", fix.rerun);
    println!();
    println!(
        "Signals: avg_top1 {:.3} | weak {} | no-match {} | expectation fails {} | dominant {:.0}% {}",
        fix.avg_top1_similarity,
        fix.low_similarity_queries,
        fix.no_match_queries,
        fix.expectation_failures,
        fix.dominant_rate * 100.0,
        fix.dominant_doc.as_deref().unwrap_or("<none>")
    );
    Ok(())
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
    use std::collections::HashSet;

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
            let mut seen_docs = HashSet::new();
            let mut top_docs = Vec::new();
            for c in r.ranked.iter().take(3) {
                let doc = crate::diagnostics::chunk_doc_id(&c.chunk_id);
                if seen_docs.insert(doc.clone()) {
                    top_docs.push(doc);
                }
            }
            (best, top_docs)
        } else {
            (None, Vec::new())
        };

        let top_doc_ranks = result.map(|r| {
            let mut seen_docs = HashSet::new();
            let mut rows = Vec::new();
            for c in r.ranked.iter().take(3) {
                let doc = crate::diagnostics::chunk_doc_id(&c.chunk_id);
                if seen_docs.insert(doc.clone()) {
                    rows.push(RankEntry { doc, rank: c.rank });
                }
            }
            rows
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

#[cfg(test)]
mod tests {
    use super::expectation_outcomes;
    use crate::model::{RankedChunk, RetrievalResult};
    use crate::retrieval::QuerySpec;

    #[test]
    fn expectation_preview_dedupes_duplicate_docs_in_top3() {
        let retrievals = vec![RetrievalResult {
            query_id: "q1".to_string(),
            query_text: "refund".to_string(),
            ranked: vec![
                RankedChunk {
                    chunk_id: "faq.md#0".to_string(),
                    score: 0.9,
                    rank: 1,
                },
                RankedChunk {
                    chunk_id: "faq.md#1".to_string(),
                    score: 0.8,
                    rank: 2,
                },
                RankedChunk {
                    chunk_id: "refund.md#0".to_string(),
                    score: 0.7,
                    rank: 3,
                },
            ],
        }];
        let queries = vec![QuerySpec {
            id: "q1".to_string(),
            query: "refund".to_string(),
            expect_docs: vec!["refund.md".to_string()],
        }];

        let rows = expectation_outcomes(&retrievals, &queries, 5);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].top_docs, vec!["faq.md", "refund.md"]);
        assert_eq!(
            rows[0]
                .top_doc_ranks
                .as_ref()
                .expect("top_doc_ranks should exist")
                .len(),
            2
        );
    }
}
