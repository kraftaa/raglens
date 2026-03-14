use crate::model::{Corpus, Finding, RetrievalResult};
use crate::retrieval::{CompareReport, ExplanationReport};
use anyhow::Result;
use serde::Serialize;

pub fn print_readiness(corpus: &Corpus, findings: &[Finding], json: bool) -> Result<()> {
    if json {
        let payload = JsonReport {
            documents: corpus.documents.len(),
            chunks: corpus.chunks.len(),
            findings,
            retrievals: &[],
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("RAG Retrieval Readiness Report");
    println!("==============================");
    println!();
    println!("Documents: {}", corpus.documents.len());
    println!("Chunks: {}", corpus.chunks.len());
    println!();
    print_findings(findings);
    Ok(())
}

pub fn print_simulation(
    corpus: &Corpus,
    retrievals: &[RetrievalResult],
    findings: &[Finding],
    json: bool,
) -> Result<()> {
    if json {
        let payload = JsonReport {
            documents: corpus.documents.len(),
            chunks: corpus.chunks.len(),
            findings,
            retrievals,
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("RAG Retrieval Simulation");
    println!("========================");
    println!();
    println!("Queries: {}", retrievals.len());
    println!("Documents: {}", corpus.documents.len());
    println!("Chunks: {}", corpus.chunks.len());
    println!();
    print_findings(findings);
    Ok(())
}

pub fn print_chunks(corpus: &Corpus, findings: &[Finding], json: bool) -> Result<()> {
    if json {
        let payload = JsonReport {
            documents: corpus.documents.len(),
            chunks: corpus.chunks.len(),
            findings,
            retrievals: &[],
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    println!("Chunk Diagnostics");
    println!("=================");
    println!();
    println!("Chunks: {}", corpus.chunks.len());
    println!();
    print_findings(findings);
    Ok(())
}

pub fn print_coverage(corpus: &Corpus, findings: &[Finding], json: bool) -> Result<()> {
    if json {
        let payload = JsonReport {
            documents: corpus.documents.len(),
            chunks: corpus.chunks.len(),
            findings,
            retrievals: &[],
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    println!("Topic Coverage");
    println!("==============");
    println!();
    print_findings(findings);
    Ok(())
}

pub fn print_explanation(query: &str, report: &ExplanationReport) -> Result<()> {
    println!("Retrieval Explanation");
    println!("=====================");
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

pub fn print_comparison(query: &str, report: &CompareReport) -> Result<()> {
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
    findings: &'a [Finding],
    #[serde(skip_serializing_if = "slice_is_empty")]
    retrievals: &'a [RetrievalResult],
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
