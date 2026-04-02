mod chunker;
mod cli;
mod compare_runs;
mod config;
mod diagnostics;
mod embeddings;
mod loader;
mod model;
mod normalize;
mod report;
mod retrieval;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, CompareFormat};
use config::Config;
use serde::Serialize;
use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Copy)]
struct FailOnOpts {
    dominant: Option<f32>,
    weak: Option<usize>,
    no_match: Option<usize>,
}

#[derive(Clone, Copy)]
struct CompareGateOpts {
    fail_if_weak_increases: bool,
    fail_if_no_match_increases: bool,
    fail_if_similarity_drops: bool,
    fail_if_regressed: bool,
    fail_if_top1_dominant_rate_exceeds: Option<f32>,
    fail_if_top1_dominant_rate_increases: bool,
    fail_if_query_count_mismatch: bool,
}

#[derive(Debug)]
struct ExitCodedError {
    code: i32,
    msg: String,
}

impl fmt::Display for ExitCodedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for ExitCodedError {}

fn fail_gate_error(msg: impl Into<String>) -> anyhow::Error {
    ExitCodedError {
        code: 2,
        msg: msg.into(),
    }
    .into()
}

fn compare_gate_error(msg: impl Into<String>) -> anyhow::Error {
    ExitCodedError {
        code: 3,
        msg: msg.into(),
    }
    .into()
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut config = Config::load(cli.config.as_deref())?;
    config.override_embedder(cli.embedder.as_deref())?;
    if let Some(cache) = cli.cache_dir {
        config.cache_dir = cache;
    }
    config.override_thresholds(cli.low_sim_threshold, cli.no_match_threshold);
    config.validate()?;
    validate_unit_interval_opt("fail_on_dominant", cli.fail_on_dominant)?;
    if let Some(dir) = &cli.artifacts_dir {
        std::fs::create_dir_all(dir)?;
    }
    let output = report::OutputOpts {
        artifacts: cli.artifacts_dir.as_ref(),
        json_out: cli.json_out.as_ref(),
        json: false,
    };
    let fail_on = FailOnOpts {
        dominant: cli.fail_on_dominant,
        weak: cli.fail_on_weak,
        no_match: cli.fail_on_no_match,
    };

    match cli.command {
        Commands::Readiness {
            path,
            queries,
            json,
        } => run_readiness(
            path,
            queries,
            &config,
            report::OutputOpts { json, ..output },
            fail_on,
        )?,
        Commands::Simulate {
            path,
            queries,
            json,
        } => run_simulate(
            path,
            queries,
            &config,
            report::OutputOpts { json, ..output },
            fail_on,
        )?,
        Commands::Chunks { path, json } => {
            ensure_fail_flags_supported("chunks", fail_on)?;
            run_chunks(path, &config, report::OutputOpts { json, ..output })?
        }
        Commands::Coverage {
            path,
            queries,
            json,
        } => {
            ensure_fail_flags_supported("coverage", fail_on)?;
            run_coverage(
                path,
                queries,
                &config,
                report::OutputOpts { json, ..output },
            )?
        }
        Commands::Explain { path, query } => {
            ensure_fail_flags_supported("explain", fail_on)?;
            run_explain(path, &config, output, query)?
        }
        Commands::CompareQuery { path, query } => {
            ensure_fail_flags_supported("compare-query", fail_on)?;
            run_compare(path, &config, output, query)?
        }
        Commands::CompareRuns {
            baseline,
            improved,
            format,
            fail_if_weak_increases,
            fail_if_no_match_increases,
            fail_if_similarity_drops,
            fail_if_regressed,
            fail_if_top1_dominant_rate_exceeds,
            fail_if_top1_dominant_rate_increases,
            fail_if_query_count_mismatch,
            json,
        } => run_compare_runs(
            {
                ensure_fail_flags_supported("compare-runs", fail_on)?;
                baseline
            },
            improved,
            format,
            CompareGateOpts {
                fail_if_weak_increases,
                fail_if_no_match_increases,
                fail_if_similarity_drops,
                fail_if_regressed,
                fail_if_top1_dominant_rate_exceeds,
                fail_if_top1_dominant_rate_increases,
                fail_if_query_count_mismatch,
            },
            report::OutputOpts { json, ..output },
        )?,
        Commands::SelfTest {
            docs,
            queries,
            json,
        } => {
            ensure_fail_flags_supported("self-test", fail_on)?;
            run_self_test(
                docs,
                queries,
                &config,
                report::OutputOpts { json, ..output },
            )?
        }
    }

    Ok(())
}

pub fn run_with_exit_code() -> i32 {
    match run() {
        Ok(()) => 0,
        Err(err) => {
            let msg = err.to_string();
            eprintln!("Error: {msg}");
            if let Some(coded) = err.downcast_ref::<ExitCodedError>() {
                coded.code
            } else {
                1
            }
        }
    }
}

fn prepare_corpus(path: PathBuf, config: &Config) -> Result<model::Corpus> {
    let docs = loader::load_documents(&path)?;
    let normalized = normalize::normalize_documents(docs);
    let chunks = chunker::chunk_documents(&normalized, config);
    Ok(model::Corpus {
        documents: normalized,
        chunks,
    })
}

fn run_readiness(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    output: report::OutputOpts<'_>,
    fail_on: FailOnOpts,
) -> Result<()> {
    if queries.is_none() && has_any_fail_flags(fail_on) {
        anyhow::bail!("--fail-on-* flags for readiness require --queries");
    }
    let corpus = prepare_corpus(path, config)?;
    let (stats, findings_chunks) = diagnostics::chunk_stats(&corpus, config);
    let mut findings = diagnostics::run_readiness(&corpus, config);
    findings.extend(findings_chunks);
    let mut sim_summary = None;
    let mut coverage = None;
    let mut sim_results: Option<Vec<crate::model::RetrievalResult>> = None;
    if let Some(qpath) = queries {
        let embedder = embeddings::build_embedder(config)?;
        let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&qpath), config)?;
        let retrieval_findings = diagnostics::analyze_retrieval(&sim.results, &sim.queries, config);
        findings.extend(retrieval_findings);
        findings.extend(diagnostics::analyze_dominant_causes(
            &corpus,
            &sim.results,
            config,
        ));
        sim_summary = Some(diagnostics::simulate_summary(
            &sim.results,
            config.low_sim_threshold,
            config.no_match_threshold,
        ));
        coverage = Some(diagnostics::coverage_summary(
            &sim.results,
            config.low_sim_threshold,
            config.no_match_threshold,
        ));
        sim_results = Some(sim.results);
    }
    report::print_readiness(
        &corpus,
        &findings,
        &stats,
        sim_summary.as_ref(),
        coverage.as_ref(),
        config,
        output,
    )?;
    if let Some(results) = sim_results.as_ref() {
        apply_fail_flags(
            fail_on.dominant,
            fail_on.weak,
            fail_on.no_match,
            &sim_summary,
            results,
            config,
        )?;
    }
    Ok(())
}

fn run_simulate(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    output: report::OutputOpts<'_>,
    fail_on: FailOnOpts,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let sim =
        retrieval::simulate_retrieval(&corpus, embedder.as_ref(), queries.as_deref(), config)?;
    let mut findings = diagnostics::analyze_retrieval(&sim.results, &sim.queries, config);
    findings.extend(diagnostics::analyze_dominant_causes(
        &corpus,
        &sim.results,
        config,
    ));
    report::print_simulation(
        &corpus,
        &sim.results,
        &sim.queries,
        config,
        &findings,
        output,
    )?;
    let summary = diagnostics::simulate_summary(
        &sim.results,
        config.low_sim_threshold,
        config.no_match_threshold,
    );
    apply_fail_flags(
        fail_on.dominant,
        fail_on.weak,
        fail_on.no_match,
        &Some(summary),
        &sim.results,
        config,
    )?;
    Ok(())
}

fn run_chunks(path: PathBuf, config: &Config, output: report::OutputOpts<'_>) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let (stats, findings) = diagnostics::chunk_stats(&corpus, config);
    report::print_chunks(&corpus, &findings, &stats, config, output)?;
    Ok(())
}

fn run_coverage(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    if let Some(qpath) = queries {
        let embedder = embeddings::build_embedder(config)?;
        let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&qpath), config)?;
        let summary = diagnostics::coverage_summary(
            &sim.results,
            config.low_sim_threshold,
            config.no_match_threshold,
        );
        report::print_coverage(&corpus, &[], Some(&summary), config, output)?;
    } else {
        let findings = diagnostics::analyze_topics(&corpus, config);
        report::print_coverage(&corpus, &findings, None, config, output)?;
    }
    Ok(())
}

fn run_explain(
    path: PathBuf,
    config: &Config,
    output: report::OutputOpts<'_>,
    query: String,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let explanation = retrieval::explain_query(&corpus, embedder.as_ref(), &query, config)?;
    report::print_explanation(&query, &explanation, output)?;
    Ok(())
}

fn run_compare(
    path: PathBuf,
    config: &Config,
    output: report::OutputOpts<'_>,
    query: String,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let explanation = retrieval::explain_query(&corpus, embedder.as_ref(), &query, config)?;
    let comparison = retrieval::CompareReport {
        ranked: explanation.ranked.into_iter().take(5).collect(),
    };
    report::print_comparison(&query, &comparison, output)?;
    Ok(())
}

fn run_compare_runs(
    baseline: PathBuf,
    improved: PathBuf,
    format: CompareFormat,
    gates: CompareGateOpts,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let diff = compare_runs::compare_runs(&baseline, &improved)?;
    if let Some(threshold) = gates.fail_if_top1_dominant_rate_exceeds {
        if !(0.0..=1.0).contains(&threshold) {
            anyhow::bail!(
                "fail_if_top1_dominant_rate_exceeds must be in [0, 1], got {}",
                threshold
            );
        }
    }
    report::print_run_comparison(&diff, format, output)?;
    if let Some(msg) = compare_gate_failure_message(&diff, gates) {
        return Err(compare_gate_error(msg));
    }
    Ok(())
}

#[derive(Serialize)]
struct SelfTestCheck {
    name: String,
    pass: bool,
    detail: String,
}

#[derive(Serialize)]
struct SelfTestReport {
    meta: SelfTestMeta,
    ok: bool,
    documents: usize,
    chunks: usize,
    queries: usize,
    checks: Vec<SelfTestCheck>,
}

#[derive(Serialize)]
struct SelfTestMeta {
    schema_version: &'static str,
    tool_version: &'static str,
    command: &'static str,
}

fn run_self_test(
    docs: PathBuf,
    queries: PathBuf,
    config: &Config,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let corpus = prepare_corpus(docs, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&queries), config)?;

    let mut checks = Vec::new();
    checks.push(SelfTestCheck {
        name: "documents_loaded".to_string(),
        pass: !corpus.documents.is_empty(),
        detail: format!("documents={}", corpus.documents.len()),
    });
    checks.push(SelfTestCheck {
        name: "chunks_produced".to_string(),
        pass: !corpus.chunks.is_empty(),
        detail: format!("chunks={}", corpus.chunks.len()),
    });
    checks.push(SelfTestCheck {
        name: "queries_loaded".to_string(),
        pass: !sim.queries.is_empty(),
        detail: format!("queries={}", sim.queries.len()),
    });

    let expectation_fails = expectation_failures(&sim.results, &sim.queries, config.top_k);
    checks.push(SelfTestCheck {
        name: "expectations".to_string(),
        pass: expectation_fails == 0,
        detail: format!("failed_expectations={}", expectation_fails),
    });

    let ok = checks.iter().all(|c| c.pass);
    let payload = SelfTestReport {
        meta: SelfTestMeta {
            schema_version: "1",
            tool_version: env!("CARGO_PKG_VERSION"),
            command: "self-test",
        },
        ok,
        documents: corpus.documents.len(),
        chunks: corpus.chunks.len(),
        queries: sim.queries.len(),
        checks,
    };

    if let Some(path) = output.json_out {
        let data = serde_json::to_vec_pretty(&payload)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, data)?;
    } else if let Some(dir) = output.artifacts {
        let path = dir.join("self_test.json");
        std::fs::create_dir_all(dir)?;
        std::fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
    }

    if output.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("Self-test");
        println!("=========");
        println!(
            "Documents: {} | Chunks: {} | Queries: {}",
            payload.documents, payload.chunks, payload.queries
        );
        for check in &payload.checks {
            println!(
                "- {}: {} ({})",
                if check.pass { "PASS" } else { "FAIL" },
                check.name,
                check.detail
            );
        }
    }

    if !ok {
        return Err(fail_gate_error("Fail: self-test checks failed"));
    }
    Ok(())
}

fn validate_unit_interval_opt(name: &str, value: Option<f32>) -> Result<()> {
    if let Some(v) = value {
        if !(0.0..=1.0).contains(&v) {
            anyhow::bail!("{name} must be in [0, 1], got {}", v);
        }
    }
    Ok(())
}

fn has_any_fail_flags(fail_on: FailOnOpts) -> bool {
    fail_on.dominant.is_some() || fail_on.weak.is_some() || fail_on.no_match.is_some()
}

fn ensure_fail_flags_supported(command: &str, fail_on: FailOnOpts) -> Result<()> {
    if has_any_fail_flags(fail_on) {
        anyhow::bail!(
            "--fail-on-* flags are supported only for readiness/simulate (not {})",
            command
        );
    }
    Ok(())
}

fn compare_gate_failure_message(
    diff: &compare_runs::SimDiff,
    gates: CompareGateOpts,
) -> Option<String> {
    if gates.fail_if_weak_increases && diff.weak_after > diff.weak_before {
        return Some(format!(
            "Fail: weak matches increased ({} -> {})",
            diff.weak_before, diff.weak_after
        ));
    }
    if gates.fail_if_no_match_increases && diff.no_match_after > diff.no_match_before {
        return Some(format!(
            "Fail: no-match queries increased ({} -> {})",
            diff.no_match_before, diff.no_match_after
        ));
    }
    if gates.fail_if_similarity_drops
        && diff.avg_top1_similarity_after + f32::EPSILON < diff.avg_top1_similarity_before
    {
        return Some(format!(
            "Fail: avg top-1 similarity dropped ({:.3} -> {:.3})",
            diff.avg_top1_similarity_before, diff.avg_top1_similarity_after
        ));
    }
    if gates.fail_if_regressed && diff.verdict == "REGRESSED" {
        return Some("Fail: compare verdict is REGRESSED".to_string());
    }
    if gates.fail_if_query_count_mismatch && diff.query_count_mismatch {
        return Some(format!(
            "Fail: query counts differ (before {} vs after {})",
            diff.queries_before, diff.queries_after
        ));
    }
    if let Some(threshold) = gates.fail_if_top1_dominant_rate_exceeds {
        if diff.top1_dominant_rate_after > threshold {
            let doc = diff
                .top1_dominant_doc_after
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string());
            return Some(format!(
                "Fail: top-1 dominant rate after is {:.3} for {} > threshold {:.3}",
                diff.top1_dominant_rate_after, doc, threshold
            ));
        }
    }
    if gates.fail_if_top1_dominant_rate_increases
        && diff.top1_dominant_rate_after > diff.top1_dominant_rate_before
    {
        let before_doc = diff
            .top1_dominant_doc_before
            .clone()
            .unwrap_or_else(|| "<none>".to_string());
        let after_doc = diff
            .top1_dominant_doc_after
            .clone()
            .unwrap_or_else(|| "<none>".to_string());
        return Some(format!(
            "Fail: top-1 dominant rate increased ({:.3} {} -> {:.3} {})",
            diff.top1_dominant_rate_before, before_doc, diff.top1_dominant_rate_after, after_doc
        ));
    }
    None
}

fn expectation_failures(
    results: &[crate::model::RetrievalResult],
    queries: &[retrieval::QuerySpec],
    top_k: usize,
) -> usize {
    let mut failures = 0usize;
    for spec in queries {
        if spec.expect_docs.is_empty() {
            continue;
        }
        let result = results.iter().find(|r| r.query_id == spec.id);
        let ok = result
            .map(|r| {
                spec.expect_docs.iter().any(|doc| {
                    r.ranked.iter().any(|chunk| {
                        diagnostics::chunk_doc_id(&chunk.chunk_id) == *doc && chunk.rank <= top_k
                    })
                })
            })
            .unwrap_or(false);
        if !ok {
            failures += 1;
        }
    }
    failures
}

fn apply_fail_flags(
    fail_on_dominant: Option<f32>,
    fail_on_weak: Option<usize>,
    fail_on_no_match: Option<usize>,
    sim_summary: &Option<crate::model::SimSummary>,
    retrievals: &[crate::model::RetrievalResult],
    config: &Config,
) -> Result<()> {
    let Some(summary) = sim_summary.as_ref() else {
        return Ok(());
    };
    if let Some(thresh) = fail_on_weak {
        if summary.low_similarity_queries > thresh {
            return Err(fail_gate_error(format!(
                "Fail: weak queries {} > threshold {} (low_sim_threshold {:.2})",
                summary.low_similarity_queries, thresh, config.low_sim_threshold
            )));
        }
    }
    if let Some(thresh) = fail_on_no_match {
        if summary.no_match_queries > thresh {
            return Err(fail_gate_error(format!(
                "Fail: no-match queries {} > threshold {} (no_match_threshold {:.2})",
                summary.no_match_queries, thresh, config.no_match_threshold
            )));
        }
    }
    if let Some(thresh) = fail_on_dominant {
        if let Some(max_rate) = diagnostics::max_dominant_rate(retrievals, config) {
            if max_rate > thresh {
                return Err(fail_gate_error(format!(
                    "Fail: dominant doc rate {:.2} > threshold {:.2}",
                    max_rate, thresh
                )));
            }
        }
    }
    Ok(())
}
