use crate::model::{RetrievedDoc, RunArtifact};
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct TraceReport {
    pub question: String,
    pub answer: String,
    pub retrieved_docs: usize,
    pub evidence: Vec<EvidenceTrace>,
    pub best_evidence: Option<EvidenceTrace>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvidenceTrace {
    pub rank: usize,
    pub doc_id: String,
    pub score: Option<f64>,
    pub source: Option<String>,
    pub chunk_id: Option<String>,
    pub vector_id: Option<String>,
    pub overlap: f64,
    pub support_hint: SupportHint,
    pub text_preview: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportHint {
    Strong,
    Weak,
    None,
}

pub fn trace_run_file(path: &Path) -> Result<TraceReport> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let run: RunArtifact =
        serde_json::from_reader(file).with_context(|| format!("parsing {}", path.display()))?;
    Ok(trace_run(&run))
}

pub fn trace_run(run: &RunArtifact) -> TraceReport {
    let mut evidence = run
        .retrieved_docs
        .iter()
        .enumerate()
        .map(|(idx, doc)| evidence_trace(idx + 1, &run.answer, doc))
        .collect::<Vec<_>>();
    evidence.sort_by(|a, b| {
        b.overlap
            .total_cmp(&a.overlap)
            .then_with(|| a.rank.cmp(&b.rank))
    });
    let best_evidence = evidence.first().cloned();

    TraceReport {
        question: run.question.clone(),
        answer: run.answer.clone(),
        retrieved_docs: run.retrieved_docs.len(),
        evidence,
        best_evidence,
    }
}

pub fn render_trace_text(report: &TraceReport) -> String {
    let mut out = String::new();
    out.push_str("RAGLens Trace\n");
    out.push_str("=============\n\n");
    out.push_str("Question:\n");
    out.push_str(&report.question);
    out.push_str("\n\nAnswer:\n");
    out.push_str(&report.answer);
    out.push_str("\n\nRetrieved docs: ");
    out.push_str(&report.retrieved_docs.to_string());
    out.push_str("\n\n");

    if let Some(best) = &report.best_evidence {
        out.push_str("Best evidence match:\n");
        out.push_str(&format_evidence_line(best));
        out.push('\n');
        out.push_str("  preview: ");
        out.push_str(&best.text_preview);
        out.push_str("\n\n");
    }

    out.push_str("Evidence chain:\n");
    if report.evidence.is_empty() {
        out.push_str("  none\n");
    } else {
        for evidence in &report.evidence {
            out.push_str(&format_evidence_line(evidence));
            out.push('\n');
        }
    }

    out
}

fn evidence_trace(rank: usize, answer: &str, doc: &RetrievedDoc) -> EvidenceTrace {
    let overlap = token_overlap(answer, &doc.text);
    EvidenceTrace {
        rank,
        doc_id: doc.id.clone(),
        score: doc.score,
        source: doc.source.clone(),
        chunk_id: metadata_string(doc, "chunk_id").or_else(|| metadata_string(doc, "chunk")),
        vector_id: metadata_string(doc, "vector_id").or_else(|| metadata_string(doc, "vector")),
        overlap,
        support_hint: support_hint(overlap),
        text_preview: preview(&doc.text, 180),
    }
}

fn support_hint(overlap: f64) -> SupportHint {
    if overlap >= 0.35 {
        SupportHint::Strong
    } else if overlap >= 0.15 {
        SupportHint::Weak
    } else {
        SupportHint::None
    }
}

fn metadata_string(doc: &RetrievedDoc, key: &str) -> Option<String> {
    doc.metadata
        .as_ref()?
        .get(key)?
        .as_str()
        .map(str::to_string)
}

fn token_overlap(left: &str, right: &str) -> f64 {
    let left_tokens = tokens(left);
    let right_tokens = tokens(right);
    if left_tokens.is_empty() {
        return 0.0;
    }
    let matches = left_tokens
        .iter()
        .filter(|token| right_tokens.contains(*token))
        .count();
    matches as f64 / left_tokens.len() as f64
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter_map(|part| {
            let token = part.trim().to_ascii_lowercase();
            (token.len() >= 3).then_some(token)
        })
        .collect()
}

fn preview(text: &str, limit: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= limit {
        compact
    } else {
        let truncated = compact.chars().take(limit).collect::<String>();
        format!("{truncated}...")
    }
}

fn format_evidence_line(evidence: &EvidenceTrace) -> String {
    let mut parts = vec![
        format!("  #{} {}", evidence.rank, evidence.doc_id),
        format!("support={}", support_hint_label(&evidence.support_hint)),
        format!("overlap={:.2}", evidence.overlap),
    ];
    if let Some(score) = evidence.score {
        parts.push(format!("score={score:.3}"));
    }
    if let Some(source) = &evidence.source {
        parts.push(format!("source={source}"));
    }
    if let Some(chunk_id) = &evidence.chunk_id {
        parts.push(format!("chunk={chunk_id}"));
    }
    if let Some(vector_id) = &evidence.vector_id {
        parts.push(format!("vector={vector_id}"));
    }
    parts.join(" | ")
}

fn support_hint_label(hint: &SupportHint) -> &'static str {
    match hint {
        SupportHint::Strong => "strong",
        SupportHint::Weak => "weak",
        SupportHint::None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RunArtifact;

    #[test]
    fn trace_selects_best_evidence_by_answer_overlap() {
        let run = RunArtifact {
            question: "Can I get a refund after 90 days?".to_string(),
            answer: "Refunds are only allowed within 30 days.".to_string(),
            retrieved_docs: vec![
                RetrievedDoc {
                    id: "shipping".to_string(),
                    text: "Shipping takes five business days.".to_string(),
                    score: Some(0.6),
                    source: Some("shipping.md".to_string()),
                    metadata: None,
                },
                RetrievedDoc {
                    id: "refund_policy".to_string(),
                    text: "Refunds are only allowed within 30 days for eligible defects."
                        .to_string(),
                    score: Some(0.9),
                    source: Some("refund.md".to_string()),
                    metadata: None,
                },
            ],
            claims: Vec::new(),
            metrics: None,
            context: None,
        };

        let report = trace_run(&run);

        assert_eq!(
            report.best_evidence.as_ref().map(|e| e.doc_id.as_str()),
            Some("refund_policy")
        );
    }
}
