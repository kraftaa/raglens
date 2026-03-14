mod chunker;
mod cli;
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
    let config = Config::default();

    match cli.command {
        Commands::Readiness { path, json } => run_readiness(path, &config, json)?,
        Commands::Simulate {
            path,
            queries,
            json,
        } => run_simulate(path, queries, &config, json)?,
        Commands::Chunks { path, json } => run_chunks(path, &config, json)?,
        Commands::Coverage { path, json } => run_coverage(path, &config, json)?,
        Commands::Explain { path, query } => run_explain(path, &config, query)?,
        Commands::Compare { path, query } => run_compare(path, &config, query)?,
    }

    Ok(())
}

fn prepare_corpus(path: PathBuf, config: &Config) -> Result<model::Corpus> {
    let docs = loader::load_documents(&path)?;
    let normalized = normalize::normalize_documents(docs);
    let chunks = chunker::chunk_documents(&normalized, config);
    Ok(model::Corpus { documents: normalized, chunks })
}

fn run_readiness(path: PathBuf, config: &Config, json: bool) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let findings = diagnostics::run_readiness(&corpus, config);
    report::print_readiness(&corpus, &findings, json)?;
    Ok(())
}

fn run_simulate(
    path: PathBuf,
    queries: Option<PathBuf>,
    config: &Config,
    json: bool,
) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::NullEmbedder::default();
    let sim =
        retrieval::simulate_retrieval(&corpus, &embedder, queries.as_deref(), config)?;
    let findings = diagnostics::analyze_retrieval(&sim.results, &sim.queries, config);
    report::print_simulation(&corpus, &sim.results, &findings, json)?;
    Ok(())
}

fn run_chunks(path: PathBuf, config: &Config, json: bool) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let findings = diagnostics::analyze_chunks(&corpus, config);
    report::print_chunks(&corpus, &findings, json)?;
    Ok(())
}

fn run_coverage(path: PathBuf, config: &Config, json: bool) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let findings = diagnostics::analyze_topics(&corpus, config);
    report::print_coverage(&corpus, &findings, json)?;
    Ok(())
}

fn run_explain(path: PathBuf, config: &Config, query: String) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::NullEmbedder::default();
    let explanation = retrieval::explain_query(&corpus, &embedder, &query, config)?;
    report::print_explanation(&query, &explanation)?;
    Ok(())
}

fn run_compare(path: PathBuf, config: &Config, query: String) -> Result<()> {
    let corpus = prepare_corpus(path, config)?;
    let embedder = embeddings::NullEmbedder::default();
    let comparison = retrieval::compare_query(&corpus, &embedder, &query, config)?;
    report::print_comparison(&query, &comparison)?;
    Ok(())
}
