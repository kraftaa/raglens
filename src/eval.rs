use crate::model::{
    EvalCaseResult, EvalIssue, EvalRegression, EvalReport, RetrievedDoc, RunArtifact,
};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RulesFile {
    List(Vec<CaseRule>),
    Wrapped { cases: Vec<CaseRule> },
}

#[derive(Debug, Clone, Deserialize)]
struct CaseRule {
    id: String,
    question: String,
    #[serde(default)]
    expected: ExpectedRule,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct ExpectedRule {
    grounded: bool,
    must_include: Vec<String>,
    must_not_include: Vec<String>,
    min_chars: Option<usize>,
    max_chars: Option<usize>,
    require_json: bool,
    json_required_keys: Vec<String>,
}

#[derive(Debug, Clone)]
struct RunCase {
    id: String,
    run: RunArtifact,
}

pub fn evaluate_run_path(run_path: &Path, rules_path: &Path) -> Result<EvalReport> {
    let rules = load_rules(rules_path)?;
    let runs = load_runs(run_path)?;

    let mut by_question: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, run) in runs.iter().enumerate() {
        by_question
            .entry(normalize_text(&run.run.question))
            .or_default()
            .push(idx);
    }

    let mut used = vec![false; runs.len()];
    let mut cases = Vec::with_capacity(rules.len());

    for rule in &rules {
        let mut picked = pick_run_by_rule_id(rule, &runs, &used);
        let key = normalize_text(&rule.question);
        if let Some(idx) = picked {
            used[idx] = true;
        }
        if picked.is_none() {
            if let Some(candidates) = by_question.get(&key) {
                for idx in candidates {
                    if !used[*idx] {
                        used[*idx] = true;
                        picked = Some(*idx);
                        break;
                    }
                }
            }
        }

        if let Some(idx) = picked {
            cases.push(evaluate_case(rule, &runs[idx].run));
        } else {
            cases.push(EvalCaseResult {
                id: rule.id.clone(),
                question: rule.question.clone(),
                pass: false,
                answer_preview: "<missing run artifact for rule>".to_string(),
                retrieved_docs: 0,
                grounded_overlap: 0.0,
                unsupported_claims: Vec::new(),
                issues: vec![EvalIssue {
                    code: "MISSING_RUN".to_string(),
                    message: format!("no run artifact matched question '{}'", rule.question),
                }],
            });
        }
    }

    let unmatched_runs = runs
        .iter()
        .enumerate()
        .filter_map(|(idx, run)| {
            if used[idx] {
                None
            } else {
                Some(run.id.clone())
            }
        })
        .collect::<Vec<_>>();

    let passed_cases = cases.iter().filter(|c| c.pass).count();
    let total_cases = cases.len();
    let failed_cases = total_cases.saturating_sub(passed_cases);
    let pass_rate = if total_cases == 0 {
        0.0
    } else {
        passed_cases as f64 / total_cases as f64
    };

    Ok(EvalReport {
        run_input: run_path.display().to_string(),
        rules_input: rules_path.display().to_string(),
        total_cases,
        passed_cases,
        failed_cases,
        pass_rate,
        cases,
        unmatched_runs,
    })
}

pub fn load_eval_report(path: &Path) -> Result<EvalReport> {
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let report: EvalReport = serde_json::from_slice(&data)
        .with_context(|| format!("parsing eval report {}", path.display()))?;
    Ok(report)
}

pub fn regression(current: &EvalReport, baseline: &EvalReport) -> EvalRegression {
    EvalRegression {
        baseline_pass_rate: baseline.pass_rate,
        current_pass_rate: current.pass_rate,
        pass_rate_delta: current.pass_rate - baseline.pass_rate,
        baseline_failed: baseline.failed_cases,
        current_failed: current.failed_cases,
        failed_delta: current.failed_cases as isize - baseline.failed_cases as isize,
    }
}

pub fn render_eval_text(report: &EvalReport) -> String {
    let mut out = String::new();
    out.push_str("RAGLens Eval\n");
    out.push_str("===========\n\n");
    out.push_str(&format!(
        "PASS: {:.0}% ({}/{})\nFAIL: {:.0}% ({}/{})\n\n",
        report.pass_rate * 100.0,
        report.passed_cases,
        report.total_cases,
        (1.0 - report.pass_rate) * 100.0,
        report.failed_cases,
        report.total_cases
    ));

    if report.failed_cases == 0 {
        out.push_str("No failing cases.\n");
    } else {
        out.push_str("Failures:\n");
        for case in report.cases.iter().filter(|c| !c.pass).take(10) {
            out.push_str(&format!("- {}:\n", case.id));
            for issue in &case.issues {
                out.push_str(&format!("  - {} {}\n", issue.code, issue.message));
            }
        }
    }

    if !report.unmatched_runs.is_empty() {
        out.push_str("\nUnmatched run artifacts:\n");
        for id in report.unmatched_runs.iter().take(10) {
            out.push_str(&format!("- {}\n", id));
        }
    }

    out
}

pub fn render_report_markdown(report: &EvalReport, baseline: Option<&EvalReport>) -> String {
    let mut out = String::new();
    out.push_str("## Summary\n");
    out.push_str(&format!(
        "- Pass rate: {:.0}% ({}/{})\n",
        report.pass_rate * 100.0,
        report.passed_cases,
        report.total_cases
    ));
    out.push_str(&format!("- Failed cases: {}\n", report.failed_cases));
    if let Some(base) = baseline {
        let reg = regression(report, base);
        out.push_str(&format!(
            "- Regression vs baseline: pass-rate delta {:+.1}pp, failed-case delta {:+}\n",
            reg.pass_rate_delta * 100.0,
            reg.failed_delta
        ));
    }

    out.push_str("\n## Top failures\n");
    let mut any = false;
    for case in report.cases.iter().filter(|c| !c.pass).take(10) {
        any = true;
        let issue = case
            .issues
            .first()
            .map(|i| format!("{} {}", i.code, i.message))
            .unwrap_or_else(|| "UNKNOWN failure".to_string());
        out.push_str(&format!("- {} -> {}\n", case.id, issue));
    }
    if !any {
        out.push_str("- none\n");
    }

    out.push_str("\n## Notes\n");
    if report.unmatched_runs.is_empty() {
        out.push_str("- all runs matched at least one rule question\n");
    } else {
        out.push_str(&format!(
            "- {} run artifacts were not matched to rules\n",
            report.unmatched_runs.len()
        ));
    }
    out
}

pub fn render_report_html(report: &EvalReport, baseline: Option<&EvalReport>) -> String {
    let markdown = render_report_markdown(report, baseline);
    let escaped = html_escape(&markdown).replace('\n', "<br/>");
    format!(
        "<!doctype html>\
<html lang=\"en\">\
<head>\
  <meta charset=\"utf-8\"/>\
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\
  <title>RAGLens Report</title>\
  <style>\
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; line-height: 1.5; }}\
    h1 {{ margin: 0 0 8px; }}\
    .content {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: #fafafa; }}\
  </style>\
</head>\
<body>\
  <h1>RAGLens Evaluation Report</h1>\
  <div class=\"content\">{}</div>\
</body>\
</html>",
        escaped
    )
}

fn load_rules(path: &Path) -> Result<Vec<CaseRule>> {
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: RulesFile = serde_yaml::from_slice(&data)
        .with_context(|| format!("parsing rules {}", path.display()))?;
    let cases = match parsed {
        RulesFile::List(cases) => cases,
        RulesFile::Wrapped { cases } => cases,
    };
    if cases.is_empty() {
        anyhow::bail!("rules file {} contains no cases", path.display());
    }
    Ok(cases)
}

fn load_runs(path: &Path) -> Result<Vec<RunCase>> {
    if path.is_file() {
        let run = load_run_file(path)?;
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("run")
            .to_string();
        return Ok(vec![RunCase { id, run }]);
    }
    if !path.is_dir() {
        anyhow::bail!("run path is neither file nor directory: {}", path.display());
    }

    let mut files = WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        anyhow::bail!(
            "no JSON files found under {}; expected run artifact JSON files",
            path.display()
        );
    }

    let mut runs = Vec::new();
    let mut first_run_like_error: Option<anyhow::Error> = None;
    for file in files {
        let data = fs::read(&file).with_context(|| format!("reading {}", file.display()))?;
        let value: Value = match serde_json::from_slice(&data) {
            Ok(v) => v,
            Err(_) => {
                // Ignore non-run JSON files in mixed artifact directories.
                continue;
            }
        };

        let run_like = value
            .as_object()
            .map(|obj| obj.contains_key("question") || obj.contains_key("answer"))
            .unwrap_or(false);
        if !run_like {
            continue;
        }

        match serde_json::from_value::<RunArtifact>(value) {
            Ok(run) => {
                if run.question.trim().is_empty() || run.answer.trim().is_empty() {
                    if first_run_like_error.is_none() {
                        first_run_like_error = Some(anyhow::anyhow!(
                            "run artifact {} must include non-empty question and answer",
                            file.display()
                        ));
                    }
                    continue;
                }
                let rel = file
                    .strip_prefix(path)
                    .ok()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| file.display().to_string());
                runs.push(RunCase { id: rel, run });
            }
            Err(err) => {
                if first_run_like_error.is_none() {
                    first_run_like_error = Some(anyhow::anyhow!(
                        "invalid run artifact {}: {}",
                        file.display(),
                        err
                    ));
                }
            }
        }
    }

    if runs.is_empty() {
        if let Some(err) = first_run_like_error {
            return Err(err);
        }
        anyhow::bail!(
            "no valid run artifacts found under {}; expected JSON files with question/answer/retrieved_docs",
            path.display()
        );
    }

    Ok(runs)
}

fn load_run_file(path: &Path) -> Result<RunArtifact> {
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let run: RunArtifact = serde_json::from_slice(&data)
        .with_context(|| format!("parsing run artifact {}", path.display()))?;
    if run.question.trim().is_empty() || run.answer.trim().is_empty() {
        anyhow::bail!(
            "run artifact {} must include non-empty question and answer",
            path.display()
        );
    }
    Ok(run)
}

fn pick_run_by_rule_id(rule: &CaseRule, runs: &[RunCase], used: &[bool]) -> Option<usize> {
    let rule_id = normalize_id(&rule.id);
    for (idx, run) in runs.iter().enumerate() {
        if used[idx] {
            continue;
        }
        let run_id = normalize_id(&run.id);
        if run_id == rule_id {
            return Some(idx);
        }
        if let Some(stem) = run_stem(&run.id) {
            if normalize_id(stem) == rule_id {
                return Some(idx);
            }
        }
    }
    None
}

fn run_stem(run_id: &str) -> Option<&str> {
    Path::new(run_id).file_stem().and_then(|s| s.to_str())
}

fn evaluate_case(rule: &CaseRule, run: &RunArtifact) -> EvalCaseResult {
    let mut issues = Vec::new();
    let answer_lower = run.answer.to_lowercase();
    let answer_preview = preview(&run.answer, 220);

    for phrase in &rule.expected.must_include {
        if !answer_lower.contains(&phrase.to_lowercase()) {
            issues.push(EvalIssue {
                code: "MISSING_REQUIRED_PHRASE".to_string(),
                message: format!("answer missing required phrase '{}'", phrase),
            });
        }
    }

    for phrase in &rule.expected.must_not_include {
        if answer_lower.contains(&phrase.to_lowercase()) {
            issues.push(EvalIssue {
                code: "FORBIDDEN_PHRASE".to_string(),
                message: format!("answer contains forbidden phrase '{}'", phrase),
            });
        }
    }

    let answer_len = run.answer.chars().count();
    if let Some(min_chars) = rule.expected.min_chars {
        if answer_len < min_chars {
            issues.push(EvalIssue {
                code: "ANSWER_TOO_SHORT".to_string(),
                message: format!("answer length below min_chars {}", min_chars),
            });
        }
    }
    if let Some(max_chars) = rule.expected.max_chars {
        if answer_len > max_chars {
            issues.push(EvalIssue {
                code: "ANSWER_TOO_LONG".to_string(),
                message: format!("answer length above max_chars {}", max_chars),
            });
        }
    }

    if rule.expected.require_json || !rule.expected.json_required_keys.is_empty() {
        match serde_json::from_str::<Value>(&run.answer) {
            Ok(value) => {
                for key in &rule.expected.json_required_keys {
                    if value.get(key).is_none() {
                        issues.push(EvalIssue {
                            code: "MISSING_JSON_KEY".to_string(),
                            message: format!("answer JSON missing required key '{}'", key),
                        });
                    }
                }
            }
            Err(_) => issues.push(EvalIssue {
                code: "INVALID_JSON_ANSWER".to_string(),
                message: "answer is not valid JSON".to_string(),
            }),
        }
    }

    let grounded_overlap = best_doc_overlap(&run.answer, &run.retrieved_docs);
    let unsupported_claims = unsupported_claims(&run.answer, &run.retrieved_docs);
    if rule.expected.grounded {
        if run.retrieved_docs.is_empty() {
            issues.push(EvalIssue {
                code: "NO_RETRIEVED_DOCS".to_string(),
                message: "grounding check requested but no retrieved documents provided"
                    .to_string(),
            });
        }
        if grounded_overlap < 0.10 {
            issues.push(EvalIssue {
                code: "LOW_GROUNDING_OVERLAP".to_string(),
                message: format!(
                    "answer-to-doc lexical overlap is low ({:.2})",
                    grounded_overlap
                ),
            });
        }
        if !unsupported_claims.is_empty() {
            issues.push(EvalIssue {
                code: "UNSUPPORTED_CLAIMS".to_string(),
                message: format!(
                    "{} answer sentence(s) lacked support in retrieved docs",
                    unsupported_claims.len()
                ),
            });
        }
    }

    EvalCaseResult {
        id: rule.id.clone(),
        question: rule.question.clone(),
        pass: issues.is_empty(),
        answer_preview,
        retrieved_docs: run.retrieved_docs.len(),
        grounded_overlap,
        unsupported_claims,
        issues,
    }
}

fn unsupported_claims(answer: &str, docs: &[RetrievedDoc]) -> Vec<String> {
    split_sentences(answer)
        .into_iter()
        .filter(|sent| token_set(sent).len() >= 4)
        .filter(|sent| best_doc_overlap(sent, docs) < 0.20)
        .collect()
}

fn best_doc_overlap(text: &str, docs: &[RetrievedDoc]) -> f64 {
    docs.iter()
        .map(|doc| lexical_overlap(text, &doc.text))
        .fold(0.0, f64::max)
}

fn lexical_overlap(a: &str, b: &str) -> f64 {
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

fn token_set(text: &str) -> HashSet<String> {
    let stop_words = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if", "in", "is",
        "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where",
        "which", "why", "with", "you", "your",
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let mut set = HashSet::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            buf.extend(ch.to_lowercase());
        } else if !buf.is_empty() {
            if !stop_words.contains(buf.as_str()) {
                set.insert(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if !buf.is_empty() && !stop_words.contains(buf.as_str()) {
        set.insert(buf);
    }
    set
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let sentence = buf.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_string());
            }
            buf.clear();
        }
    }
    let tail = buf.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

fn normalize_text(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn normalize_id(input: &str) -> String {
    let mut normalized = String::new();
    for ch in input.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
        }
    }
    if normalized.is_empty() {
        input.trim().to_lowercase()
    } else {
        normalized
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[derive(serde::Serialize)]
pub struct ReportPayload {
    pub eval: EvalReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regression: Option<EvalRegression>,
}

pub fn load_report_or_eval(run_path: &Path, rules_path: Option<&Path>) -> Result<EvalReport> {
    if let Some(rules) = rules_path {
        return evaluate_run_path(run_path, rules);
    }
    load_eval_report(run_path).with_context(|| {
        format!(
            "run input {} is not an eval report JSON; pass --rules to evaluate raw run artifacts",
            run_path.display()
        )
    })
}
