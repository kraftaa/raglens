use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// RAG retrieval diagnostics CLI
#[derive(Parser, Debug)]
#[command(name = "rag-audit", version, about = "RAG retrieval readiness diagnostics", long_about = None)]
pub struct Cli {
    /// Optional config file (defaults to rag-audit.toml in CWD)
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    /// Override embedder provider: null | openai
    #[arg(long, global = true)]
    pub embedder: Option<String>,
    /// Override cache directory
    #[arg(long, global = true)]
    pub cache_dir: Option<PathBuf>,
    /// Override low similarity threshold (default from config)
    #[arg(long, global = true)]
    pub low_sim_threshold: Option<f32>,
    /// Override no-match threshold (default from config)
    #[arg(long, global = true)]
    pub no_match_threshold: Option<f32>,
    /// Write JSON artifacts to this directory (e.g., artifacts/)
    #[arg(long, global = true)]
    pub artifacts_dir: Option<PathBuf>,
    /// Write single JSON output to this file (overrides artifacts_dir for that run)
    #[arg(long, global = true)]
    pub json_out: Option<PathBuf>,
    /// Fail if dominant document rate exceeds this value (0-1)
    #[arg(long, global = true)]
    pub fail_on_dominant: Option<f32>,
    /// Fail if weak (low similarity) queries exceed this count
    #[arg(long, global = true)]
    pub fail_on_weak: Option<usize>,
    /// Fail if no-match queries exceed this count
    #[arg(long, global = true)]
    pub fail_on_no_match: Option<usize>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Full retrieval readiness audit
    Readiness {
        /// Path to documents directory
        path: PathBuf,
        /// Optional queries file (yaml or plain text) to include retrieval coverage/dominance
        #[arg(long)]
        queries: Option<PathBuf>,
        /// Output JSON instead of human report
        #[arg(long)]
        json: bool,
    },
    /// Simulate retrieval with synthetic or user queries
    Simulate {
        path: PathBuf,
        /// Optional YAML queries file
        #[arg(long)]
        queries: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Chunk size and coherence diagnostics
    Chunks {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Topic coverage/imbalance detection
    Coverage {
        path: PathBuf,
        /// Optional queries file for coverage evaluation
        #[arg(long)]
        queries: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Explain why top docs ranked for a query
    #[command(name = "explain")]
    Explain {
        path: PathBuf,
        #[arg(long)]
        query: String,
    },
    /// Compare near-match docs/chunks for a query
    #[command(name = "compare-query", alias = "compare")]
    CompareQuery {
        path: PathBuf,
        #[arg(long)]
        query: String,
    },
    /// Compare two simulation JSON reports (before/after)
    #[command(name = "compare-runs", alias = "compare-sim")]
    CompareRuns {
        /// Baseline simulation JSON
        baseline: PathBuf,
        /// Improved simulation JSON
        improved: PathBuf,
        /// Output format: summary or table
        #[arg(long, value_enum, default_value_t = CompareFormat::Summary)]
        format: CompareFormat,
        /// Output JSON diff instead of human table
        #[arg(long)]
        json: bool,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum CompareFormat {
    Summary,
    Table,
}
