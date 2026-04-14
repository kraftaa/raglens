mod answer_audit;
mod chunker;
mod cli;
mod compare_runs;
mod config;
mod diagnostics;
mod embeddings;
mod loader;
mod mcp_import;
mod model;
mod normalize;
mod report;
mod retrieval;
mod run_diff;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, CompareFormat, DiffOutputFormat, PeriodGranularity};
use config::Config;
use model::{FixReport, OptimizeCandidate, OptimizeSummary};
use serde::Serialize;
use std::fmt;
use std::fs;
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

struct OptimizeOpts {
    path: PathBuf,
    queries: PathBuf,
    chunk_sizes: Vec<usize>,
    chunk_overlaps: Vec<usize>,
    top_n: usize,
    write_config: Option<PathBuf>,
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
        html_out: cli.html_out.as_ref(),
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
        Commands::Diff {
            baseline,
            current,
            format,
        } => {
            ensure_fail_flags_supported("diff", fail_on)?;
            run_diff(baseline, current, format, output)?
        }
        Commands::SaveRun {
            out,
            question,
            answer,
            retrieved_docs,
            model,
            top_k,
        } => {
            ensure_fail_flags_supported("save-run", fail_on)?;
            run_save_run(out, question, answer, retrieved_docs, model, top_k)?
        }
        Commands::McpImport {
            input,
            out,
            question_pointer,
            answer_pointer,
            docs_pointer,
            model,
            top_k,
        } => {
            ensure_fail_flags_supported("mcp-import", fail_on)?;
            run_mcp_import(
                input,
                out,
                question_pointer,
                answer_pointer,
                docs_pointer,
                model,
                top_k,
            )?
        }
        Commands::Fix {
            path,
            queries,
            json,
        } => {
            ensure_fail_flags_supported("fix", fail_on)?;
            run_fix(
                path,
                queries,
                &config,
                report::OutputOpts { json, ..output },
            )?
        }
        Commands::AnswerAudit {
            data,
            auto,
            period_granularity,
            group_by,
            metric,
            period_col,
            baseline,
            current,
            question,
            answer,
            weak_contribution_threshold,
            json,
        } => {
            ensure_fail_flags_supported("answer-audit", fail_on)?;
            run_answer_audit(
                answer_audit::AnswerAuditInput {
                    data,
                    auto,
                    period_granularity: match period_granularity {
                        PeriodGranularity::Raw => "raw".to_string(),
                        PeriodGranularity::Month => "month".to_string(),
                        PeriodGranularity::Week => "week".to_string(),
                    },
                    group_by,
                    metric,
                    period_col,
                    baseline,
                    current,
                    question,
                    answer,
                    weak_contribution_threshold,
                },
                report::OutputOpts { json, ..output },
            )?
        }
        Commands::CompareQuery { path, query } => {
            ensure_fail_flags_supported("compare-query", fail_on)?;
            run_compare(path, &config, output, query)?
        }
        Commands::Optimize {
            path,
            queries,
            chunk_sizes,
            chunk_overlaps,
            top_n,
            write_config,
            json,
        } => {
            ensure_fail_flags_supported("optimize", fail_on)?;
            run_optimize(
                OptimizeOpts {
                    path,
                    queries,
                    chunk_sizes,
                    chunk_overlaps,
                    top_n,
                    write_config,
                },
                &config,
                report::OutputOpts { json, ..output },
            )?
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
    queries: PathBuf,
    config: &Config,
    output: report::OutputOpts<'_>,
    fail_on: FailOnOpts,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&queries), config)?;
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

fn run_fix(
    path: PathBuf,
    queries: PathBuf,
    config: &Config,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let docs_path = path.clone();
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&queries), config)?;
    let summary = diagnostics::simulate_summary(
        &sim.results,
        config.low_sim_threshold,
        config.no_match_threshold,
    );
    let (chunk_stats, _) = diagnostics::chunk_stats(&corpus, config);
    let expectation_fails = expectation_failures(&sim.results, &sim.queries, config.top_k);

    let dominant = summary
        .top1_freq
        .iter()
        .max_by(|a, b| a.count.cmp(&b.count).then_with(|| b.doc_id.cmp(&a.doc_id)))
        .map(|d| {
            (
                d.doc_id.clone(),
                d.count as f32 / (summary.queries.max(1) as f32),
            )
        });
    let dominant_doc = dominant.as_ref().map(|d| d.0.clone());
    let dominant_rate = dominant.as_ref().map(|d| d.1).unwrap_or(0.0);

    let mut issue = String::new();
    let mut likely_causes = Vec::new();
    let mut first_fix = String::new();

    let tiny_ratio = if chunk_stats.chunks == 0 {
        0.0
    } else {
        chunk_stats.small_chunks as f32 / chunk_stats.chunks as f32
    };

    if let Some((doc, rate)) = dominant {
        if rate >= config.dominant_threshold {
            issue = format!("{} dominates {:.0}% of top-1 results", doc, rate * 100.0);
            if chunk_stats.p95_tokens > ((config.chunk_size as f32) * 1.4) as usize {
                likely_causes.push("chunk size is too large for mixed-topic content".to_string());
            }
            if chunk_stats.duplicate_chunks > 0 {
                likely_causes
                    .push("duplicate/repeated chunk language boosts one document".to_string());
            }
            if likely_causes.is_empty() {
                likely_causes.push(
                    "one broad document is matching many queries more than specific docs"
                        .to_string(),
                );
            }
            let suggested = suggest_smaller_chunk_size(config, 0.5);
            first_fix = format_reduce_chunk_size_fix(config, suggested);
        }
    }

    if issue.is_empty() && chunk_stats.large_chunks > 0 {
        issue = format!(
            "{} oversized chunks detected (p95 tokens {})",
            chunk_stats.large_chunks, chunk_stats.p95_tokens
        );
        likely_causes.push("chunk boundary is too coarse for document structure".to_string());
        let suggested = suggest_smaller_chunk_size(config, 0.7);
        first_fix = format_reduce_chunk_size_fix(config, suggested);
    }

    if issue.is_empty() && tiny_ratio >= 0.35 {
        issue = format!("{:.0}% of chunks are too small", tiny_ratio * 100.0);
        likely_causes.push("splitting is too aggressive for this corpus".to_string());
        let suggested = ((config.chunk_size as f32) * 1.4).round() as usize;
        first_fix = format!(
            "increase chunk_size from {} to {}",
            config.chunk_size, suggested
        );
    }

    if issue.is_empty() && expectation_fails > 0 {
        issue = format!("{expectation_fails} expected documents are missing in top-k");
        likely_causes.push("relevant passages are split or diluted across chunks".to_string());
        likely_causes.push("expected docs may be under-represented lexically".to_string());
        let suggested = suggest_smaller_chunk_size(config, 0.5);
        first_fix = if suggested < config.chunk_size {
            format!(
                "try smaller chunks first: chunk_size {} -> {}",
                config.chunk_size, suggested
            )
        } else {
            format!(
                "chunk_size is already near minimum for overlap {}; decrease chunk_overlap first",
                config.chunk_overlap
            )
        };
    }

    if issue.is_empty() && summary.no_match_queries > 0 {
        issue = format!(
            "{} queries have no reliable match",
            summary.no_match_queries
        );
        likely_causes.push("content for those intents may be missing in the corpus".to_string());
        likely_causes.push("query wording may not align with document wording".to_string());
        first_fix =
            "add/expand source docs for missed intents and include exact user phrasing".to_string();
    }

    if issue.is_empty() && summary.low_similarity_queries > 0 {
        issue = format!(
            "{} queries are low similarity",
            summary.low_similarity_queries
        );
        likely_causes.push("retrieval language mismatch between queries and docs".to_string());
        first_fix =
            "add examples/aliases in docs to match how users phrase these queries".to_string();
    }

    if issue.is_empty() {
        issue = "no major retrieval issue detected in current sample".to_string();
        likely_causes.push(
            "current chunking and retrieval settings are stable for tested queries".to_string(),
        );
        first_fix = "keep current settings; add harder queries before tuning".to_string();
    }

    let rerun = format!(
        "raglens simulate {} --queries {}",
        docs_path.display(),
        queries.display()
    );
    let report = FixReport {
        issue,
        likely_causes,
        first_fix,
        rerun,
        avg_top1_similarity: summary.avg_top1_similarity,
        low_similarity_queries: summary.low_similarity_queries,
        no_match_queries: summary.no_match_queries,
        expectation_failures: expectation_fails,
        dominant_doc,
        dominant_rate,
    };

    report::print_fix(&report, output)?;
    Ok(())
}

fn run_answer_audit(
    input: answer_audit::AnswerAuditInput,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let (req, auto_notes) = answer_audit::resolve_input(input)?;
    let report = answer_audit::audit_answer(&req)?;

    #[derive(Serialize)]
    struct AnswerAuditJson<'a> {
        meta: AnswerAuditMeta,
        auto_notes: &'a [String],
        audit: &'a answer_audit::AnswerAuditReport,
    }

    #[derive(Serialize)]
    struct AnswerAuditMeta {
        schema_version: &'static str,
        tool_version: &'static str,
        command: &'static str,
    }

    let payload = AnswerAuditJson {
        meta: AnswerAuditMeta {
            schema_version: "1",
            tool_version: env!("CARGO_PKG_VERSION"),
            command: "answer-audit",
        },
        auto_notes: &auto_notes,
        audit: &report,
    };

    write_json_artifact(output, "answer_audit.json", &payload)?;

    if output.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("AI Answer Audit");
    println!("===============");
    println!();
    if !auto_notes.is_empty() {
        println!("Resolved config:");
        for note in &auto_notes {
            println!("- {}", note);
        }
        println!();
    }
    if let Some(question) = &report.question {
        println!("Question: {}", question);
    }
    println!("Answer: {}", report.answer);
    println!();
    println!("Verdict: {}", report.verdict);
    println!(
        "Periods: baseline '{}' -> current '{}'",
        report.baseline_period, report.current_period
    );
    println!("Total delta: {:+.2}", report.total_delta);
    println!();

    println!("Top Drivers:");
    for d in &report.top_drivers {
        println!(
            "- {}: {:+.2} ({:+.1}%)",
            d.label, d.delta, d.contribution_pct
        );
    }
    println!();

    if report.issues.is_empty() {
        println!("Issues: none");
    } else {
        println!("Issues:");
        for issue in &report.issues {
            println!("- {}: {}", issue.code, issue.message);
        }
    }
    println!();
    println!(
        "Coverage: {:.0}% | Alignment score: {:.2} | Confidence: {}",
        report.coverage_pct, report.alignment_score, report.confidence
    );
    Ok(())
}

fn run_diff(
    baseline: PathBuf,
    current: PathBuf,
    format: DiffOutputFormat,
    output: report::OutputOpts<'_>,
) -> Result<()> {
    let report = run_diff::compare_run_files(&baseline, &current)?;
    write_json_artifact(output, "diff.json", &report)?;

    if output.json || format == DiffOutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("{}", run_diff::render_diff_text(&report));
    Ok(())
}

fn run_save_run(
    out: PathBuf,
    question: String,
    answer: String,
    retrieved_docs_path: Option<PathBuf>,
    model: Option<String>,
    top_k: Option<usize>,
) -> Result<()> {
    let retrieved_docs = if let Some(path) = retrieved_docs_path {
        load_retrieved_docs(&path)?
    } else {
        Vec::new()
    };

    let run = model::RunArtifact {
        question,
        answer,
        retrieved_docs,
        claims: Vec::new(),
        metrics: None,
        context: if model.is_some() || top_k.is_some() {
            Some(model::RunContext { model, top_k })
        } else {
            None
        },
    };
    run_diff::validate_artifact_for_save(&run, "--out artifact")?;

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&run)?)?;
    println!("Saved run artifact: {}", out.display());
    Ok(())
}

fn run_mcp_import(
    input: PathBuf,
    out: PathBuf,
    question_pointer: Option<String>,
    answer_pointer: Option<String>,
    docs_pointer: Option<String>,
    model: Option<String>,
    top_k: Option<usize>,
) -> Result<()> {
    let run = mcp_import::import_file(
        &input,
        &mcp_import::McpImportOpts {
            question_pointer: question_pointer.as_deref(),
            answer_pointer: answer_pointer.as_deref(),
            docs_pointer: docs_pointer.as_deref(),
            model,
            top_k,
        },
    )?;
    run_diff::validate_artifact_for_save(&run, "--out artifact")?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&run)?)?;
    println!("Saved run artifact from MCP JSON: {}", out.display());
    Ok(())
}

fn load_retrieved_docs(path: &std::path::Path) -> Result<Vec<model::RetrievedDoc>> {
    let data = fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&data)?;
    if value.is_array() {
        return Ok(serde_json::from_value(value)?);
    }
    if let Some(items) = value.get("retrieved_docs") {
        return Ok(serde_json::from_value(items.clone())?);
    }
    anyhow::bail!("retrieved docs JSON must be an array or contain a 'retrieved_docs' array field");
}

fn write_json_artifact<T: Serialize>(
    output: report::OutputOpts<'_>,
    filename: &str,
    payload: &T,
) -> Result<()> {
    if let Some(path) = output.json_out {
        let data = serde_json::to_vec_pretty(payload)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data)?;
        return Ok(());
    }
    if let Some(dir) = output.artifacts {
        let data = serde_json::to_vec_pretty(payload)?;
        fs::create_dir_all(dir)?;
        fs::write(dir.join(filename), data)?;
    }
    Ok(())
}

fn suggest_smaller_chunk_size(config: &Config, ratio: f32) -> usize {
    let min_allowed = config.chunk_overlap + 1;
    if config.chunk_size <= min_allowed {
        return config.chunk_size;
    }
    let target = ((config.chunk_size as f32) * ratio).round() as usize;
    target.clamp(min_allowed, config.chunk_size - 1)
}

fn format_reduce_chunk_size_fix(config: &Config, suggested: usize) -> String {
    if suggested < config.chunk_size {
        format!(
            "reduce chunk_size from {} to {}",
            config.chunk_size, suggested
        )
    } else {
        format!(
            "chunk_size is already near minimum for overlap {}; decrease chunk_overlap first",
            config.chunk_overlap
        )
    }
}

fn run_optimize(opts: OptimizeOpts, config: &Config, output: report::OutputOpts<'_>) -> Result<()> {
    let OptimizeOpts {
        path,
        queries,
        mut chunk_sizes,
        mut chunk_overlaps,
        top_n,
        write_config,
    } = opts;

    if chunk_sizes.is_empty() {
        anyhow::bail!("optimize requires at least one chunk size");
    }
    if chunk_overlaps.is_empty() {
        anyhow::bail!("optimize requires at least one chunk overlap");
    }
    if top_n == 0 {
        anyhow::bail!("top_n must be > 0");
    }

    chunk_sizes.sort_unstable();
    chunk_sizes.dedup();
    chunk_overlaps.sort_unstable();
    chunk_overlaps.dedup();

    let embedder = embeddings::build_embedder(config)?;
    let mut candidates = Vec::<OptimizeCandidate>::new();
    let mut skipped = 0usize;

    for size in chunk_sizes {
        for overlap in &chunk_overlaps {
            if *overlap >= size {
                skipped += 1;
                continue;
            }
            let mut cfg = config.clone();
            cfg.chunk_size = size;
            cfg.chunk_overlap = *overlap;
            cfg.validate()?;

            let corpus = prepare_corpus(path.clone(), &cfg)?;
            let sim =
                retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&queries), &cfg)?;
            let summary = diagnostics::simulate_summary(
                &sim.results,
                cfg.low_sim_threshold,
                cfg.no_match_threshold,
            );
            let expectation_fails = expectation_failures(&sim.results, &sim.queries, cfg.top_k);
            let dominant_rate = diagnostics::max_dominant_rate(&sim.results, &cfg).unwrap_or(0.0);
            let (chunk_stats, _) = diagnostics::chunk_stats(&corpus, &cfg);

            let chunk_issue_ratio = if chunk_stats.chunks == 0 {
                0.0
            } else {
                (chunk_stats.large_chunks + chunk_stats.small_chunks) as f32
                    / chunk_stats.chunks as f32
            };
            let dominance_penalty = if dominant_rate > cfg.dominant_threshold {
                (dominant_rate - cfg.dominant_threshold) * 40.0
            } else {
                0.0
            };
            let score = (summary.avg_top1_similarity * 100.0)
                - (summary.low_similarity_queries as f32 * 2.0)
                - (summary.no_match_queries as f32 * 4.0)
                - (expectation_fails as f32 * 5.0)
                - (chunk_issue_ratio * 10.0)
                - dominance_penalty;

            candidates.push(OptimizeCandidate {
                chunk_size: size,
                chunk_overlap: *overlap,
                score,
                avg_top1_similarity: summary.avg_top1_similarity,
                low_similarity_queries: summary.low_similarity_queries,
                no_match_queries: summary.no_match_queries,
                expectation_failures: expectation_fails,
                dominant_rate,
                documents: corpus.documents.len(),
                chunks: corpus.chunks.len(),
                large_chunks: chunk_stats.large_chunks,
                small_chunks: chunk_stats.small_chunks,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.expectation_failures.cmp(&b.expectation_failures))
            .then_with(|| a.no_match_queries.cmp(&b.no_match_queries))
            .then_with(|| a.low_similarity_queries.cmp(&b.low_similarity_queries))
            .then_with(|| a.chunk_size.cmp(&b.chunk_size))
            .then_with(|| a.chunk_overlap.cmp(&b.chunk_overlap))
    });

    let best = candidates.first().cloned();
    let summary = OptimizeSummary {
        queries_file: queries.display().to_string(),
        considered: candidates.len(),
        skipped,
        top_n,
        candidates,
        best: best.clone(),
    };

    if let Some(path) = write_config {
        if let Some(best_cfg) = best {
            let content = format!(
                "# generated by raglens optimize\nchunk_size = {}\nchunk_overlap = {}\n",
                best_cfg.chunk_size, best_cfg.chunk_overlap
            );
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, content)?;
        }
    }

    report::print_optimize(&summary, output)?;
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

#[cfg(test)]
mod tests {
    use super::{format_reduce_chunk_size_fix, suggest_smaller_chunk_size};
    use crate::config::Config;

    #[test]
    fn suggested_smaller_chunk_size_never_increases() {
        let cfg = Config {
            chunk_size: 80,
            chunk_overlap: 20,
            ..Config::default()
        };
        let suggested = suggest_smaller_chunk_size(&cfg, 0.5);
        assert!(suggested < cfg.chunk_size);
        assert!(suggested > cfg.chunk_overlap);
    }

    #[test]
    fn reduce_fix_message_matches_direction() {
        let cfg = Config {
            chunk_size: 80,
            chunk_overlap: 20,
            ..Config::default()
        };
        let suggested = suggest_smaller_chunk_size(&cfg, 0.5);
        let msg = format_reduce_chunk_size_fix(&cfg, suggested);
        assert!(msg.contains("reduce chunk_size"));
    }
}
