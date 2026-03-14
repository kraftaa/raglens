use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// RAG retrieval diagnostics CLI
#[derive(Parser, Debug)]
#[command(name = "rag-audit", version, about = "RAG retrieval readiness diagnostics", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Full retrieval readiness audit
    Readiness {
        /// Path to documents directory
        path: PathBuf,
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
