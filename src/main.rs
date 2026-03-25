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
use cli::{Cli, Commands};
use config::Config;
use std::path::PathBuf;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut config = Config::load(cli.config.as_deref())?;
    config.override_embedder(cli.embedder.as_deref())?;
    if let Some(cache) = cli.cache_dir {
        config.cache_dir = cache;
    }
    config.override_thresholds(cli.low_sim_threshold, cli.no_match_threshold);
    if let Some(dir) = &cli.artifacts_dir {
        std::fs::create_dir_all(dir)?;
    }
    let artifacts = cli.artifacts_dir.as_ref();
    let json_out = cli.json_out.as_ref();

    match cli.command {
        Commands::Readiness {
            path,
            queries,
            json,
        } => run_readiness(path, queries, &config, artifacts, json_out, json)?,
        Commands::Simulate {
            path,
            queries,
            json,
        } => run_simulate(path, queries, &config, artifacts, json_out, json)?,
        Commands::Chunks { path, json } => run_chunks(path, &config, artifacts, json_out, json)?,
        Commands::Coverage {
            path,
            queries,
            json,
        } => run_coverage(path, queries, &config, artifacts, json_out, json)?,
        Commands::Explain { path, query } => {
            run_explain(path, &config, artifacts, json_out, query)?
        }
        Commands::CompareQuery { path, query } => {
            run_compare(path, &config, artifacts, json_out, query)?
        }
        Commands::CompareRuns {
            baseline,
            improved,
            json,
        } => run_compare_runs(baseline, improved, artifacts, json_out, json)?,
    }

    Ok(())
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
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let (stats, findings_chunks) = diagnostics::chunk_stats(&corpus, config);
    let mut findings = diagnostics::run_readiness(&corpus, config);
    findings.extend(findings_chunks);
    let mut sim_summary = None;
    let mut coverage = None;
    if let Some(qpath) = queries {
        let embedder = embeddings::build_embedder(config)?;
        let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&qpath), config)?;
        let retrieval_findings = diagnostics::analyze_retrieval(&sim.results, &sim.queries, config);
        findings.extend(retrieval_findings);
        sim_summary = Some(diagnostics::simulate_summary(&sim.results, 0.35, 0.25));
        coverage = Some(diagnostics::coverage_summary(&sim.results));
    }
    report::print_readiness(
        &corpus,
        &findings,
        &stats,
        sim_summary.as_ref(),
        coverage.as_ref(),
        config,
        artifacts,
        json_out,
        json,
    )?;
    Ok(())
}

fn run_simulate(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let sim =
        retrieval::simulate_retrieval(&corpus, embedder.as_ref(), queries.as_deref(), config)?;
    let findings = diagnostics::analyze_retrieval(&sim.results, &sim.queries, config);
    report::print_simulation(
        &corpus,
        &sim.results,
        &sim.queries,
        config,
        &findings,
        artifacts,
        json_out,
        json,
    )?;
    Ok(())
}

fn run_chunks(
    path: PathBuf,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let (stats, findings) = diagnostics::chunk_stats(&corpus, config);
    report::print_chunks(&corpus, &findings, &stats, artifacts, json_out, json)?;
    Ok(())
}

fn run_coverage(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    if let Some(qpath) = queries {
        let embedder = embeddings::build_embedder(config)?;
        let sim = retrieval::simulate_retrieval(&corpus, embedder.as_ref(), Some(&qpath), config)?;
        let summary = diagnostics::coverage_summary(&sim.results);
        report::print_coverage(
            &corpus,
            &[],
            Some(&summary),
            config,
            artifacts,
            json_out,
            json,
        )?;
    } else {
        let findings = diagnostics::analyze_topics(&corpus, config);
        report::print_coverage(&corpus, &findings, None, config, artifacts, json_out, json)?;
    }
    Ok(())
}

fn run_explain(
    path: PathBuf,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    query: String,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let explanation = retrieval::explain_query(&corpus, embedder.as_ref(), &query, config)?;
    report::print_explanation(&query, &explanation, artifacts, json_out)?;
    Ok(())
}

fn run_compare(
    path: PathBuf,
    config: &Config,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    query: String,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::build_embedder(config)?;
    let comparison = retrieval::compare_query(&corpus, embedder.as_ref(), &query, config)?;
    report::print_comparison(&query, &comparison, artifacts, json_out)?;
    Ok(())
}

fn run_compare_runs(
    baseline: PathBuf,
    improved: PathBuf,
    artifacts: Option<&PathBuf>,
    json_out: Option<&PathBuf>,
    json: bool,
) -> Result<()> {
    let diff = compare_runs::compare_runs(&baseline, &improved)?;
    report::print_run_comparison(&diff, artifacts, json_out, json)?;
    Ok(())
}
