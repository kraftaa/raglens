use crate::model::{
    AlignmentResult, AnswerChangeType, AnswerDiff, CommonDocDiff, Confidence, DiffReport,
    RetrievalDiff, RetrievedDoc, RootCause, RunArtifact,
};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::path::Path;

pub const SCORE_SHIFT_THRESHOLD: f64 = 0.15;

pub fn compare_run_files(baseline_path: &Path, current_path: &Path) -> Result<DiffReport> {
    let baseline = load_run_artifact(baseline_path)?;
    let current = load_run_artifact(current_path)?;
    Ok(compare_runs(&baseline, &current))
}

pub(crate) fn validate_artifact_for_save(run: &RunArtifact, origin: &str) -> Result<()> {
    validate_run_artifact(run, origin)
}

fn load_run_artifact(path: &Path) -> Result<RunArtifact> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let run: RunArtifact = serde_json::from_reader(file)
        .map_err(|e| anyhow::anyhow!("parsing {}: {}", path.display(), e))?;
    validate_run_artifact(&run, &path.display().to_string())?;
    Ok(run)
}

pub fn compare_runs(baseline: &RunArtifact, current: &RunArtifact) -> DiffReport {
    let answer_diff = build_answer_diff(&baseline.answer, &current.answer);
    let retrieval_diff = build_retrieval_diff(&baseline.retrieved_docs, &current.retrieved_docs);
    let alignment = build_alignment(&baseline.answer, &current.answer, baseline, current);
    let root_cause = classify_root_cause(answer_diff.changed, retrieval_diff.changed);
    let (confidence, confidence_reason) = classify_confidence(
        baseline,
        current,
        &answer_diff,
        &retrieval_diff,
        alignment.as_ref(),
        &root_cause,
    );

    DiffReport {
        question: baseline.question.clone(),
        answer_diff,
        retrieval_diff,
        alignment,
        root_cause,
        confidence,
        confidence_reason,
    }
}

pub fn render_diff_text(report: &DiffReport) -> String {
    let mut out = String::new();
    out.push_str("RAGLens Diff\n");
    out.push_str("===========\n\n");
    out.push_str("Question:\n");
    out.push_str(&report.question);
    out.push_str("\n\n");

    out.push_str("Answer changed:\n");
    if report.answer_diff.changed {
        out.push_str("- Baseline: ");
        out.push_str(&report.answer_diff.baseline_answer);
        out.push('\n');
        out.push_str("- Current:  ");
        out.push_str(&report.answer_diff.current_answer);
        out.push_str("\n\n");
    } else {
        out.push_str("- No (answer text is unchanged)\n\n");
    }
    out.push_str("Change type:\n");
    out.push_str("- ");
    out.push_str(answer_change_type_display(&report.answer_diff.change_type));
    out.push_str("\n\n");

    out.push_str("Retrieval changed:\n");
    if report.retrieval_diff.changed {
        if report.retrieval_diff.added_docs.is_empty() {
            out.push_str("- Added top docs: none\n");
        } else {
            out.push_str("- Added top docs:\n");
            for doc in &report.retrieval_diff.added_docs {
                out.push_str("  + ");
                out.push_str(&doc.id);
                out.push_str(&format_doc_tail(doc));
                out.push('\n');
            }
        }
        if report.retrieval_diff.removed_docs.is_empty() {
            out.push_str("- Removed top docs: none\n");
        } else {
            out.push_str("- Removed top docs:\n");
            for doc in &report.retrieval_diff.removed_docs {
                out.push_str("  - ");
                out.push_str(&doc.id);
                out.push_str(&format_doc_tail(doc));
                out.push('\n');
            }
        }
        if report.retrieval_diff.added_docs.is_empty()
            && report.retrieval_diff.removed_docs.is_empty()
        {
            let mut shifted = report
                .retrieval_diff
                .common_docs
                .iter()
                .filter_map(|d| d.score_delta.map(|delta| (d, delta)))
                .filter(|(_, delta)| delta.abs() > SCORE_SHIFT_THRESHOLD)
                .collect::<Vec<_>>();
            shifted.sort_by(|a, b| {
                b.1.abs()
                    .total_cmp(&a.1.abs())
                    .then_with(|| a.0.id.cmp(&b.0.id))
            });
            if !shifted.is_empty() {
                out.push_str("- Score-shifted common docs:\n");
                for (doc, delta) in shifted.into_iter().take(3) {
                    out.push_str(&format!(
                        "  * {}: {} -> {} (Δ {:+.3})\n",
                        doc.id,
                        format_opt_score(doc.baseline_score),
                        format_opt_score(doc.current_score),
                        delta
                    ));
                }
            }
        }
    } else {
        out.push_str("- No meaningful retrieval changes detected\n");
    }
    out.push('\n');

    if let Some(aln) = &report.alignment {
        out.push_str("Evidence alignment:\n");
        out.push_str("- Baseline answer best matches: ");
        out.push_str(aln.baseline_best_doc_id.as_deref().unwrap_or("<none>"));
        out.push_str(&format!(" (overlap {:.2})", aln.baseline_overlap));
        out.push('\n');
        out.push_str("- Current answer best matches: ");
        out.push_str(aln.current_best_doc_id.as_deref().unwrap_or("<none>"));
        out.push_str(&format!(" (overlap {:.2})", aln.current_overlap));
        out.push_str("\n\n");
    }

    out.push_str("Likely root cause:\n");
    out.push_str("- ");
    out.push_str(&root_cause_message(report));
    out.push_str("\n\n");

    out.push_str("Confidence: ");
    out.push_str(match report.confidence {
        Confidence::High => "HIGH",
        Confidence::Medium => "MEDIUM",
        Confidence::Low => "LOW",
    });
    out.push('\n');
    out.push_str("Reason: ");
    out.push_str(&report.confidence_reason);
    out.push('\n');
    out
}

fn build_answer_diff(baseline_answer: &str, current_answer: &str) -> AnswerDiff {
    let change_type = classify_answer_change_type(baseline_answer, current_answer);
    AnswerDiff {
        changed: !matches!(change_type, AnswerChangeType::Equivalent),
        change_type,
        baseline_answer: baseline_answer.to_string(),
        current_answer: current_answer.to_string(),
    }
}

fn classify_answer_change_type(baseline_answer: &str, current_answer: &str) -> AnswerChangeType {
    if normalize_answer(baseline_answer) == normalize_answer(current_answer) {
        return AnswerChangeType::Equivalent;
    }

    let baseline_dir = infer_direction(baseline_answer);
    let current_dir = infer_direction(current_answer);
    if is_contradictory_direction(baseline_dir, current_dir) {
        return AnswerChangeType::Contradictory;
    }

    let baseline_tokens = token_set(baseline_answer);
    let current_tokens = token_set(current_answer);
    if baseline_tokens.is_empty() || current_tokens.is_empty() {
        return AnswerChangeType::EmphasisShift;
    }
    let common = baseline_tokens
        .iter()
        .filter(|tok| current_tokens.contains(*tok))
        .count();
    let coverage_of_baseline = common as f64 / baseline_tokens.len() as f64;
    if coverage_of_baseline >= 0.70 && current_tokens.len() > baseline_tokens.len() {
        return AnswerChangeType::MoreSpecific;
    }
    AnswerChangeType::EmphasisShift
}

fn build_retrieval_diff(baseline: &[RetrievedDoc], current: &[RetrievedDoc]) -> RetrievalDiff {
    let baseline_map = map_docs_by_id(baseline);
    let current_map = map_docs_by_id(current);

    let mut added_docs = current_map
        .iter()
        .filter(|(id, _)| !baseline_map.contains_key(*id))
        .map(|(_, doc)| (*doc).clone())
        .collect::<Vec<_>>();
    let mut removed_docs = baseline_map
        .iter()
        .filter(|(id, _)| !current_map.contains_key(*id))
        .map(|(_, doc)| (*doc).clone())
        .collect::<Vec<_>>();
    added_docs.sort_by(|a, b| a.id.cmp(&b.id));
    removed_docs.sort_by(|a, b| a.id.cmp(&b.id));

    let mut common_docs = baseline_map
        .iter()
        .filter_map(|(id, base_doc)| current_map.get(id).map(|cur_doc| (id, *base_doc, *cur_doc)))
        .map(|(id, base_doc, cur_doc)| CommonDocDiff {
            id: id.clone(),
            baseline_score: base_doc.score,
            current_score: cur_doc.score,
            score_delta: match (base_doc.score, cur_doc.score) {
                (Some(b), Some(c)) => Some(c - b),
                _ => None,
            },
        })
        .collect::<Vec<_>>();
    common_docs.sort_by(|a, b| a.id.cmp(&b.id));

    let score_shifted = common_docs
        .iter()
        .filter_map(|d| d.score_delta)
        .any(|delta| delta.abs() > SCORE_SHIFT_THRESHOLD);
    let changed = !added_docs.is_empty() || !removed_docs.is_empty() || score_shifted;

    RetrievalDiff {
        added_docs,
        removed_docs,
        common_docs,
        changed,
    }
}

fn build_alignment(
    baseline_answer: &str,
    current_answer: &str,
    baseline: &RunArtifact,
    current: &RunArtifact,
) -> Option<AlignmentResult> {
    if baseline.retrieved_docs.is_empty() && current.retrieved_docs.is_empty() {
        return None;
    }

    let (baseline_best_doc_id, baseline_overlap) =
        best_overlap_doc(baseline_answer, &baseline.retrieved_docs)
            .map(|(id, overlap)| (Some(id), overlap))
            .unwrap_or((None, 0.0));

    let (current_best_doc_id, current_overlap) =
        best_overlap_doc(current_answer, &current.retrieved_docs)
            .map(|(id, overlap)| (Some(id), overlap))
            .unwrap_or((None, 0.0));

    Some(AlignmentResult {
        baseline_best_doc_id,
        current_best_doc_id,
        baseline_overlap,
        current_overlap,
    })
}

fn classify_root_cause(answer_changed: bool, retrieval_changed: bool) -> RootCause {
    match (answer_changed, retrieval_changed) {
        (false, false) => RootCause::NoMeaningfulChange,
        (true, false) => RootCause::AnswerGenerationChanged,
        (true, true) => RootCause::RetrievalChanged,
        (false, true) => RootCause::RetrievalChangedButAnswerStable,
    }
}

fn classify_confidence(
    baseline: &RunArtifact,
    current: &RunArtifact,
    answer_diff: &AnswerDiff,
    retrieval_diff: &RetrievalDiff,
    alignment: Option<&AlignmentResult>,
    root_cause: &RootCause,
) -> (Confidence, String) {
    let same_question = normalize_answer(&baseline.question) == normalize_answer(&current.question);
    if !same_question {
        return (
            Confidence::Low,
            "baseline and current questions differ, so run-to-run causality is weak".to_string(),
        );
    }

    match root_cause {
        RootCause::NoMeaningfulChange => (
            Confidence::High,
            "answer and retrieval remained stable across runs".to_string(),
        ),
        RootCause::AnswerGenerationChanged => (
            Confidence::Medium,
            if matches!(answer_diff.change_type, AnswerChangeType::Contradictory) {
                "answer became contradictory while retrieved documents stayed stable".to_string()
            } else {
                "answer changed while retrieved documents stayed stable".to_string()
            },
        ),
        RootCause::RetrievalChanged => {
            if !retrieval_diff.added_docs.is_empty() || !retrieval_diff.removed_docs.is_empty() {
                if let Some(aln) = alignment {
                    let added_ids = retrieval_diff
                        .added_docs
                        .iter()
                        .map(|d| d.id.as_str())
                        .collect::<HashSet<_>>();
                    if aln
                        .current_best_doc_id
                        .as_deref()
                        .map(|id| added_ids.contains(id))
                        .unwrap_or(false)
                        && aln.current_overlap + 1e-9 >= aln.baseline_overlap
                    {
                        let doc = aln.current_best_doc_id.as_deref().unwrap_or("<unknown>");
                        return (
                            Confidence::High,
                            format!(
                                "retrieval changed materially and newly added doc '{}' best aligns with current answer",
                                doc
                            ),
                        );
                    }
                }
                (
                    Confidence::Medium,
                    if matches!(
                        answer_diff.change_type,
                        AnswerChangeType::EmphasisShift | AnswerChangeType::MoreSpecific
                    ) {
                        "Retrieval changed materially, but both answers remain topically consistent, indicating a shift in emphasis rather than a contradiction."
                            .to_string()
                    } else {
                        "retrieval changed materially, but alignment signal is mixed".to_string()
                    },
                )
            } else {
                (
                    Confidence::Low,
                    "retrieval change is only score-level without document swaps".to_string(),
                )
            }
        }
        RootCause::RetrievalChangedButAnswerStable => {
            if !retrieval_diff.added_docs.is_empty() || !retrieval_diff.removed_docs.is_empty() {
                (
                    Confidence::Medium,
                    "retrieval changed, but answer stayed topically consistent".to_string(),
                )
            } else {
                (
                    Confidence::Low,
                    "answer is stable and retrieval changed only by score shifts".to_string(),
                )
            }
        }
    }
}

fn map_docs_by_id(docs: &[RetrievedDoc]) -> BTreeMap<String, &RetrievedDoc> {
    let mut map = BTreeMap::new();
    for doc in docs {
        map.insert(doc.id.clone(), doc);
    }
    map
}

fn normalize_answer(text: &str) -> String {
    text.split_whitespace()
        .map(|s| s.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn best_overlap_doc(answer: &str, docs: &[RetrievedDoc]) -> Option<(String, f64)> {
    if docs.is_empty() {
        return None;
    }
    let mut best: Option<(String, f64)> = None;
    for doc in docs {
        let score = lexical_overlap_score(answer, &doc_search_text(doc));
        match &best {
            None => best = Some((doc.id.clone(), score)),
            Some((best_id, best_score)) => {
                let should_take = score > *best_score + f64::EPSILON
                    || ((score - *best_score).abs() <= f64::EPSILON && doc.id < *best_id);
                if should_take {
                    best = Some((doc.id.clone(), score));
                }
            }
        }
    }
    best
}

fn lexical_overlap_score(a: &str, b: &str) -> f64 {
    let a_tokens = token_set(a);
    let b_tokens = token_set(b);
    if a_tokens.is_empty() {
        return 0.0;
    }
    let overlap = a_tokens
        .iter()
        .filter(|tok| b_tokens.contains(*tok))
        .count();
    overlap as f64 / a_tokens.len() as f64
}

fn doc_search_text(doc: &RetrievedDoc) -> String {
    let mut parts = vec![doc.text.clone(), doc.id.clone()];
    if let Some(source) = &doc.source {
        parts.push(source.clone());
    }
    if let Some(meta) = &doc.metadata {
        for (k, v) in meta {
            parts.push(k.clone());
            if let Some(s) = v.as_str() {
                parts.push(s.to_string());
            }
        }
    }
    parts.join(" ")
}

fn token_set(text: &str) -> HashSet<String> {
    let stop = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "for", "from", "has",
        "have", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
        "were", "what", "why", "with",
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let mut set = HashSet::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            buf.extend(ch.to_lowercase());
        } else if !buf.is_empty() {
            if !stop.contains(buf.as_str()) {
                set.insert(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if !buf.is_empty() && !stop.contains(buf.as_str()) {
        set.insert(buf);
    }
    set
}

fn root_cause_message(report: &DiffReport) -> String {
    match report.root_cause {
        RootCause::NoMeaningfulChange => {
            "No meaningful answer or retrieval change detected.".to_string()
        }
        RootCause::AnswerGenerationChanged => {
            "Answer changed while retrieval stayed stable; generation behavior likely changed."
                .to_string()
        }
        RootCause::RetrievalChanged => {
            if let Some(aln) = &report.alignment {
                let added_ids = report
                    .retrieval_diff
                    .added_docs
                    .iter()
                    .map(|d| d.id.as_str())
                    .collect::<HashSet<_>>();
                if aln
                    .current_best_doc_id
                    .as_deref()
                    .map(|id| added_ids.contains(id))
                    .unwrap_or(false)
                {
                    return format!(
                        "The addition of '{}' likely introduced more specific reporting guidance into the answer.",
                        aln.current_best_doc_id.as_deref().unwrap_or("<unknown>")
                    );
                }
            }
            if let Some((doc_id, overlap)) = best_overlap_doc(
                &report.answer_diff.current_answer,
                &report.retrieval_diff.added_docs,
            ) {
                if overlap >= 0.15 {
                    let current_best = report
                        .alignment
                        .as_ref()
                        .and_then(|a| a.current_best_doc_id.as_deref())
                        .unwrap_or("<none>");
                    return format!(
                        "The addition of '{}' likely introduced more specific reporting guidance into the answer, even though '{}' remains the closest lexical match.",
                        doc_id, current_best
                    );
                }
            }
            "Document-level retrieval changes likely caused the observed answer drift.".to_string()
        }
        RootCause::RetrievalChangedButAnswerStable => {
            "Retrieval changed, but answer text stayed stable.".to_string()
        }
    }
}

fn answer_change_type_display(change_type: &AnswerChangeType) -> &'static str {
    match change_type {
        AnswerChangeType::Equivalent => "equivalent",
        AnswerChangeType::EmphasisShift => "emphasis shift (non-contradictory)",
        AnswerChangeType::MoreSpecific => "more specific (non-contradictory)",
        AnswerChangeType::Contradictory => "contradictory",
    }
}

fn validate_run_artifact(run: &RunArtifact, origin: &str) -> Result<()> {
    if run.question.trim().is_empty() {
        anyhow::bail!(
            "invalid run artifact {}: 'question' must not be empty",
            origin
        );
    }
    if run.answer.trim().is_empty() {
        anyhow::bail!(
            "invalid run artifact {}: 'answer' must not be empty",
            origin
        );
    }
    let mut seen = HashSet::new();
    for doc in &run.retrieved_docs {
        if doc.id.trim().is_empty() {
            anyhow::bail!(
                "invalid run artifact {}: retrieved_docs contains empty 'id'",
                origin
            );
        }
        if !seen.insert(doc.id.clone()) {
            anyhow::bail!(
                "invalid run artifact {}: duplicate retrieved_docs id '{}'",
                origin,
                doc.id
            );
        }
    }
    Ok(())
}

fn format_opt_score(score: Option<f64>) -> String {
    score
        .map(|s| format!("{:.3}", s))
        .unwrap_or_else(|| "n/a".to_string())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClaimDirection {
    Positive,
    Negative,
    Unknown,
}

fn infer_direction(text: &str) -> ClaimDirection {
    let toks = token_set(text);
    let has_pos = ["increase", "increased", "growth", "grew", "higher", "up"]
        .iter()
        .any(|w| toks.contains(*w));
    let has_neg = [
        "decrease",
        "decreased",
        "decline",
        "declined",
        "lower",
        "down",
        "drop",
        "dropped",
        "fell",
    ]
    .iter()
    .any(|w| toks.contains(*w));
    match (has_pos, has_neg) {
        (true, false) => ClaimDirection::Positive,
        (false, true) => ClaimDirection::Negative,
        _ => ClaimDirection::Unknown,
    }
}

fn is_contradictory_direction(a: ClaimDirection, b: ClaimDirection) -> bool {
    (a == ClaimDirection::Positive && b == ClaimDirection::Negative)
        || (a == ClaimDirection::Negative && b == ClaimDirection::Positive)
}

fn format_doc_tail(doc: &RetrievedDoc) -> String {
    let score = doc
        .score
        .map(|s| format!(" (score {:.2})", s))
        .unwrap_or_default();
    let text = summarize_text(&doc.text, 120);
    if text.is_empty() {
        score
    } else {
        format!("{score} \"{text}\"")
    }
}

fn summarize_text(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        return normalized;
    }
    let mut out = String::new();
    for ch in normalized.chars().take(max_chars.saturating_sub(3)) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::build_answer_diff;

    #[test]
    fn answer_diff_ignores_case_and_whitespace() {
        let diff = build_answer_diff(
            "Revenue increased due to US",
            " revenue   increased DUE to us ",
        );
        assert!(!diff.changed);
    }
}
