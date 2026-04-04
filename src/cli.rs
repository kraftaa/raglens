use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// RAG retrieval diagnostics CLI
#[derive(Parser, Debug)]
#[command(
    name = "raglens",
    version,
    about = "RAGLens: debug retrieval before you blame the model",
    long_about = None
)]
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
    /// Write HTML report output to this file (supported by explain/compare-query)
    #[arg(long, global = true)]
    pub html_out: Option<PathBuf>,
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
    #[command(name = "compare-query")]
    CompareQuery {
        path: PathBuf,
        #[arg(long)]
        query: String,
    },
    /// Search chunking configs and suggest best retrieval metrics
    Optimize {
        /// Path to documents directory
        path: PathBuf,
        /// Query file (yaml/plain/structured text)
        #[arg(long)]
        queries: PathBuf,
        /// Comma-separated chunk sizes to test
        #[arg(long, value_delimiter = ',', default_value = "200,300,400,600")]
        chunk_sizes: Vec<usize>,
        /// Comma-separated chunk overlaps to test
        #[arg(long, value_delimiter = ',', default_value = "20,40,80")]
        chunk_overlaps: Vec<usize>,
        /// Show top N candidates in human output
        #[arg(long, default_value_t = 5)]
        top_n: usize,
        /// Optional path to write best config snippet (.toml)
        #[arg(long)]
        write_config: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Compare two simulation JSON reports (before/after)
    #[command(name = "compare-runs", alias = "compare", alias = "compare-sim")]
    CompareRuns {
        /// Baseline simulation JSON
        baseline: PathBuf,
        /// Improved simulation JSON
        improved: PathBuf,
        /// Output format: summary or table
        #[arg(long, value_enum, default_value_t = CompareFormat::Summary)]
        format: CompareFormat,
        /// Exit non-zero if weak match count increases
        #[arg(long, default_value_t = false)]
        fail_if_weak_increases: bool,
        /// Exit non-zero if no-match count increases
        #[arg(long, default_value_t = false)]
        fail_if_no_match_increases: bool,
        /// Exit non-zero if average top-1 similarity decreases
        #[arg(long, default_value_t = false)]
        fail_if_similarity_drops: bool,
        /// Exit non-zero if overall compare verdict is REGRESSED
        #[arg(long, default_value_t = false)]
        fail_if_regressed: bool,
        /// Exit non-zero if after-run top-1 dominant doc rate exceeds threshold (0-1)
        #[arg(long)]
        fail_if_top1_dominant_rate_exceeds: Option<f32>,
        /// Exit non-zero if after-run top-1 dominant doc rate increases vs baseline
        #[arg(long, default_value_t = false)]
        fail_if_top1_dominant_rate_increases: bool,
        /// Exit non-zero if baseline/improved query counts differ
        #[arg(long, default_value_t = false)]
        fail_if_query_count_mismatch: bool,
        /// Output JSON diff instead of human table
        #[arg(long)]
        json: bool,
    },
    /// Run built-in smoke checks on a corpus and query set
    #[command(name = "self-test")]
    SelfTest {
        /// Path to documents directory
        #[arg(long, default_value = "examples/docs")]
        docs: PathBuf,
        /// Path to structured query file
        #[arg(long, default_value = "examples/queries_structured.txt")]
        queries: PathBuf,
        /// Output JSON report
        #[arg(long)]
        json: bool,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum CompareFormat {
    Summary,
    Table,
}
