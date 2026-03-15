use clap::{Parser, Subcommand};
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
    /// Write JSON artifacts to this directory (e.g., artifacts/)
    #[arg(long, global = true)]
    pub artifacts_dir: Option<PathBuf>,
    /// Write single JSON output to this file (overrides artifacts_dir for that run)
    #[arg(long, global = true)]
    pub json_out: Option<PathBuf>,

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
    Explain {
        path: PathBuf,
        #[arg(long)]
        query: String,
    },
    /// Compare near-match docs/chunks for a query
    Compare {
        path: PathBuf,
        #[arg(long)]
        query: String,
    },
}
