use assert_cmd::Command;
use predicates::str::contains;
use serde_json::json;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn fixture_path(p: &str) -> PathBuf {
    PathBuf::from("tests/fixtures").join(p)
}

#[test]
fn readiness_reports_documents_and_chunks() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("readiness")
        .arg(fixture_path("docs"))
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let v: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(v["documents"], 3);
    assert_eq!(v["chunks"], 3);
    assert_eq!(v["meta"]["schema_version"], "1");
}

#[test]
fn simulate_honors_expectations() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("simulate")
        .arg(fixture_path("docs"))
        .arg("--queries")
        .arg(fixture_path("queries.yaml"))
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let v: Value = serde_json::from_slice(&output).unwrap();
    let findings = v["findings"].as_array().cloned().unwrap_or_default();
    // No expectation misses
    let has_fail = findings.iter().any(|f| f["severity"] == "Fail");
    assert!(
        !has_fail,
        "expected no expectation failures, got {findings:?}"
    );
}

#[test]
fn raglens_alias_binary_works() {
    let mut cmd = Command::cargo_bin("raglens").unwrap();
    cmd.arg("--help").assert().success();
}

#[test]
fn compare_runs_fail_if_similarity_drops() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.80,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.60,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-similarity-drops")
        .assert()
        .failure()
        .stderr(contains("avg top-1 similarity dropped"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_weak_increases() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_weak_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.70,
                "low_similarity_queries": 2,
                "no_match_queries": 1,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.72,
                "low_similarity_queries": 5,
                "no_match_queries": 1,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-weak-increases")
        .assert()
        .failure()
        .stderr(contains("weak matches increased"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_no_match_increases() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_nomatch_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.70,
                "low_similarity_queries": 2,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.73,
                "low_similarity_queries": 2,
                "no_match_queries": 2,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-no-match-increases")
        .assert()
        .failure()
        .stderr(contains("no-match queries increased"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_regressed() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_regressed_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.80,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.60,
                "low_similarity_queries": 5,
                "no_match_queries": 3,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-regressed")
        .assert()
        .failure()
        .stderr(contains("compare verdict is REGRESSED"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_json_contains_verdict() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_json_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.50,
                "low_similarity_queries": 6,
                "no_match_queries": 2,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.60,
                "low_similarity_queries": 4,
                "no_match_queries": 1,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(v["verdict"], "IMPROVED");
    assert_eq!(v["meta"]["command"], "compare-runs");

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_top1_dominant_rate_exceeds() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_dom_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.60,
                "low_similarity_queries": 4,
                "no_match_queries": 1,
                "top1_freq": [{"doc_id":"faq.md","count":5}]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.62,
                "low_similarity_queries": 3,
                "no_match_queries": 1,
                "top1_freq": [{"doc_id":"faq.md","count":8}]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-top1-dominant-rate-exceeds")
        .arg("0.70")
        .assert()
        .failure()
        .stderr(contains("top-1 dominant rate after is"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_top1_dominant_rate_increases() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_dom_inc_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");

    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.60,
                "low_similarity_queries": 4,
                "no_match_queries": 1,
                "top1_freq": [{"doc_id":"faq.md","count":5}]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.61,
                "low_similarity_queries": 4,
                "no_match_queries": 1,
                "top1_freq": [{"doc_id":"faq.md","count":7}]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-top1-dominant-rate-increases")
        .assert()
        .failure()
        .stderr(contains("top-1 dominant rate increased"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_rejects_invalid_dominant_threshold() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_badthreshold_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");
    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.6,
                "low_similarity_queries": 3,
                "no_match_queries": 1,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.6,
                "low_similarity_queries": 3,
                "no_match_queries": 1,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-top1-dominant-rate-exceeds")
        .arg("1.5")
        .assert()
        .failure()
        .stderr(contains("must be in [0, 1]"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn readiness_rejects_invalid_fail_on_dominant() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("--fail-on-dominant")
        .arg("1.2")
        .arg("readiness")
        .arg(fixture_path("docs"))
        .assert()
        .failure()
        .stderr(contains("fail_on_dominant must be in [0, 1]"));
}

#[test]
fn readiness_json_is_emitted_before_fail_gates() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("--fail-on-dominant")
        .arg("0.0")
        .arg("readiness")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .arg("--json")
        .assert()
        .failure()
        .get_output()
        .stdout
        .clone();

    let v: Value = serde_json::from_slice(&output).unwrap();
    assert!(v["documents"].as_u64().unwrap_or(0) > 0);
}

#[test]
fn readiness_fail_flags_require_queries() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("--fail-on-weak")
        .arg("0")
        .arg("readiness")
        .arg(fixture_path("docs"))
        .assert()
        .failure()
        .stderr(contains("require --queries"));
}

#[test]
fn compare_runs_rejects_readiness_fail_flags() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_badglobal_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");
    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 1,
                "avg_top1_similarity": 0.5,
                "low_similarity_queries": 0,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 1,
                "avg_top1_similarity": 0.5,
                "low_similarity_queries": 0,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("--fail-on-dominant")
        .arg("0.2")
        .arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .assert()
        .failure()
        .stderr(contains("supported only for readiness/simulate"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn self_test_passes_on_examples() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("self-test")
        .arg("--docs")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .assert()
        .success();
}

#[test]
fn self_test_json_contains_ok() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("self-test")
        .arg("--docs")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["meta"]["command"], "self-test");
}

#[test]
fn compare_gate_exit_code_is_3() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_code3_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");
    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.8,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": [{"doc_id":"faq.md","count":2}]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.5,
                "low_similarity_queries": 5,
                "no_match_queries": 3,
                "top1_freq": [{"doc_id":"faq.md","count":8}]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-regressed")
        .assert()
        .code(3);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn compare_runs_fail_if_query_count_mismatch() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rag_audit_compare_qcount_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let improved = dir.join("improved.json");
    fs::write(
        &baseline,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 10,
                "avg_top1_similarity": 0.8,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        &improved,
        serde_json::to_vec_pretty(&json!({
            "sim_summary": {
                "queries": 8,
                "avg_top1_similarity": 0.7,
                "low_similarity_queries": 1,
                "no_match_queries": 0,
                "top1_freq": []
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("compare-runs")
        .arg(&baseline)
        .arg(&improved)
        .arg("--fail-if-query-count-mismatch")
        .assert()
        .code(3)
        .stderr(contains("query counts differ"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn readiness_gate_exit_code_is_2() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    // Use dominant gate with threshold 0 for a stable deterministic failure:
    // any non-empty retrieval run has some top-1 document with rate > 0.
    cmd.arg("--fail-on-dominant")
        .arg("0.0")
        .arg("readiness")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .assert()
        .code(2);
}

#[test]
fn explain_writes_html_output() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_path = std::env::temp_dir().join(format!("raglens_explain_{stamp}.html"));

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("explain")
        .arg("examples/docs")
        .arg("--query")
        .arg("refund after 90 days")
        .arg("--html-out")
        .arg(&out_path)
        .assert()
        .success();

    let html = fs::read_to_string(&out_path).unwrap();
    assert!(html.contains("RAGLens Retrieval Explanation"));
    assert!(html.contains("refund after 90 days"));

    let _ = fs::remove_file(out_path);
}

#[test]
fn optimize_returns_best_candidate_json() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("optimize")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .arg("--chunk-sizes")
        .arg("80,120")
        .arg("--chunk-overlaps")
        .arg("20,40")
        .arg("--top-n")
        .arg("2")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let v: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(v["meta"]["command"], "optimize");
    assert!(v["optimize"]["considered"].as_u64().unwrap_or(0) > 0);
    assert!(v["optimize"]["best"].is_object());
}

#[test]
fn fix_returns_advice_json() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let output = cmd
        .arg("fix")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let v: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(v["meta"]["command"], "fix");
    assert!(!v["fix"]["issue"].as_str().unwrap_or("").is_empty());
    assert!(!v["fix"]["first_fix"].as_str().unwrap_or("").is_empty());
}

#[test]
fn help_shows_mvp_commands_and_hides_advanced_commands() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let txt = String::from_utf8_lossy(&out);
    assert!(txt.contains("explain"));
    assert!(txt.contains("simulate"));
    assert!(txt.contains("fix"));
    assert!(!txt.contains("readiness"));
    assert!(!txt.contains("compare-runs"));
    assert!(!txt.contains("optimize"));
}

#[test]
fn answer_audit_detects_contradiction_from_example_csv() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("answer-audit")
        .arg("--data")
        .arg("inputs/examples/answer_audit/sales.csv")
        .arg("--group-by")
        .arg("region,channel")
        .arg("--metric")
        .arg("revenue")
        .arg("--period-col")
        .arg("period")
        .arg("--baseline")
        .arg("old")
        .arg("--current")
        .arg("new")
        .arg("--question")
        .arg("Why did revenue increase?")
        .arg("--answer")
        .arg("Revenue increased due to EU growth")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["meta"]["command"], "answer-audit");
    assert_eq!(v["audit"]["verdict"], "INCORRECT");
}

#[test]
fn answer_audit_supported_example() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("answer-audit")
        .arg("--data")
        .arg("inputs/examples/answer_audit/sales_supported.csv")
        .arg("--group-by")
        .arg("region,channel")
        .arg("--metric")
        .arg("revenue")
        .arg("--period-col")
        .arg("period")
        .arg("--baseline")
        .arg("old")
        .arg("--current")
        .arg("new")
        .arg("--answer")
        .arg("Revenue increased due to strong US Direct growth")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["audit"]["verdict"], "SUPPORTED");
}

#[test]
fn answer_audit_risky_example() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("answer-audit")
        .arg("--data")
        .arg("inputs/examples/answer_audit/sales_risky.csv")
        .arg("--group-by")
        .arg("region,channel")
        .arg("--metric")
        .arg("revenue")
        .arg("--period-col")
        .arg("period")
        .arg("--baseline")
        .arg("old")
        .arg("--current")
        .arg("new")
        .arg("--answer")
        .arg("Revenue increased due to US Direct and LATAM growth")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["audit"]["verdict"], "RISKY");
}

#[test]
fn answer_audit_auto_infers_schema() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("answer-audit")
        .arg("--data")
        .arg("inputs/examples/answer_audit/sales.csv")
        .arg("--auto")
        .arg("--answer")
        .arg("Revenue increased due to EU growth")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["audit"]["verdict"], "INCORRECT");
    assert!(
        v["auto_notes"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "auto_notes should include inferred fields"
    );
}

#[test]
fn answer_audit_month_granularity_buckets_dates() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_answer_audit_month_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let csv_path = dir.join("sales.csv");
    fs::write(
        &csv_path,
        "region,channel,date,revenue\nUS,Direct,2026-01-03,100\nUS,Direct,2026-01-28,110\nUS,Direct,2026-02-02,260\nEU,Partner,2026-01-12,80\nEU,Partner,2026-02-14,70\n",
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("answer-audit")
        .arg("--data")
        .arg(&csv_path)
        .arg("--auto")
        .arg("--period-granularity")
        .arg("month")
        .arg("--answer")
        .arg("Revenue increased due to US Direct growth")
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["audit"]["baseline_period"], "2026-01");
    assert_eq!(v["audit"]["current_period"], "2026-02");
    assert_eq!(v["audit"]["verdict"], "SUPPORTED");
    let _ = fs::remove_dir_all(dir);
}

fn write_temp_json(path: &std::path::Path, value: &serde_json::Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

#[test]
fn diff_case_a_no_meaningful_change() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_diff_a_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let current = dir.join("current.json");
    let run = json!({
        "question": "Why did revenue increase?",
        "answer": "Revenue increased due to US growth",
        "retrieved_docs": [
            {"id":"doc_us","text":"US revenue increased by 40.","score":0.91}
        ]
    });
    write_temp_json(&baseline, &run);
    write_temp_json(&current, &run);

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("diff")
        .arg("--baseline")
        .arg(&baseline)
        .arg("--current")
        .arg(&current)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["answer_diff"]["changed"], false);
    assert_eq!(v["retrieval_diff"]["changed"], false);
    assert_eq!(v["root_cause"], "no_meaningful_change");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn diff_case_b_answer_generation_changed() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_diff_b_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let current = dir.join("current.json");
    write_temp_json(
        &baseline,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to EU growth",
            "retrieved_docs": [
                {"id":"doc_eu","text":"EU revenue increased by 20.","score":0.88}
            ]
        }),
    );
    write_temp_json(
        &current,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US Direct growth",
            "retrieved_docs": [
                {"id":"doc_eu","text":"EU revenue increased by 20.","score":0.88}
            ]
        }),
    );

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("diff")
        .arg("--baseline")
        .arg(&baseline)
        .arg("--current")
        .arg(&current)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["answer_diff"]["changed"], true);
    assert_eq!(v["retrieval_diff"]["changed"], false);
    assert_eq!(v["root_cause"], "answer_generation_changed");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn diff_case_c_retrieval_changed_with_answer_change() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_diff_c_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let current = dir.join("current.json");
    write_temp_json(
        &baseline,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to EU growth",
            "retrieved_docs": [
                {"id":"doc_eu","text":"EU Partner revenue increased by 20.","score":0.88}
            ]
        }),
    );
    write_temp_json(
        &current,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US Direct growth",
            "retrieved_docs": [
                {"id":"doc_us","text":"US Direct revenue increased by 40.","score":0.91}
            ]
        }),
    );

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("diff")
        .arg("--baseline")
        .arg(&baseline)
        .arg("--current")
        .arg(&current)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["answer_diff"]["changed"], true);
    assert_eq!(v["retrieval_diff"]["changed"], true);
    assert_eq!(v["root_cause"], "retrieval_changed");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn diff_case_d_retrieval_changed_but_answer_stable() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_diff_d_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let current = dir.join("current.json");
    write_temp_json(
        &baseline,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US growth",
            "retrieved_docs": [
                {"id":"doc_old","text":"Legacy doc.","score":0.40}
            ]
        }),
    );
    write_temp_json(
        &current,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US growth",
            "retrieved_docs": [
                {"id":"doc_new","text":"Updated doc.","score":0.90}
            ]
        }),
    );

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("diff")
        .arg("--baseline")
        .arg(&baseline)
        .arg("--current")
        .arg(&current)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["answer_diff"]["changed"], false);
    assert_eq!(v["retrieval_diff"]["changed"], true);
    assert_eq!(v["root_cause"], "retrieval_changed_but_answer_stable");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn diff_case_e_missing_scores_supported() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_diff_e_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let baseline = dir.join("baseline.json");
    let current = dir.join("current.json");
    write_temp_json(
        &baseline,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US growth",
            "retrieved_docs": [
                {"id":"doc_old","text":"Old doc without score"}
            ]
        }),
    );
    write_temp_json(
        &current,
        &json!({
            "question": "Why did revenue increase?",
            "answer": "Revenue increased due to US growth",
            "retrieved_docs": [
                {"id":"doc_new","text":"New doc without score"}
            ]
        }),
    );

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    let out = cmd
        .arg("diff")
        .arg("--baseline")
        .arg(&baseline)
        .arg("--current")
        .arg(&current)
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["retrieval_diff"]["changed"], true);
    assert!(v["retrieval_diff"]["added_docs"].as_array().is_some());
    assert!(v["retrieval_diff"]["removed_docs"].as_array().is_some());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn save_run_writes_artifact_from_retrieved_docs_array() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_save_run_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("run.json");
    let docs = dir.join("retrieved.json");
    fs::write(
        &docs,
        serde_json::to_vec_pretty(&json!([
            {
                "id": "doc_1",
                "text": "US Direct revenue increased by 40.",
                "score": 0.91
            }
        ]))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("save-run")
        .arg("--out")
        .arg(&out)
        .arg("--question")
        .arg("Why did revenue increase?")
        .arg("--answer")
        .arg("Revenue increased due to US growth")
        .arg("--retrieved-docs")
        .arg(&docs)
        .arg("--model")
        .arg("gpt-4.1")
        .arg("--top-k")
        .arg("3")
        .assert()
        .success();

    let written: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(written["question"], "Why did revenue increase?");
    assert_eq!(written["context"]["model"], "gpt-4.1");
    assert_eq!(written["context"]["top_k"], 3);
    assert_eq!(written["retrieved_docs"][0]["id"], "doc_1");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn save_run_rejects_duplicate_retrieved_doc_ids() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_save_run_dup_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("run.json");
    let docs = dir.join("retrieved.json");
    fs::write(
        &docs,
        serde_json::to_vec_pretty(&json!([
            {"id":"doc_1","text":"first"},
            {"id":"doc_1","text":"duplicate"}
        ]))
        .unwrap(),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("save-run")
        .arg("--out")
        .arg(&out)
        .arg("--question")
        .arg("Why did revenue increase?")
        .arg("--answer")
        .arg("Revenue increased due to US growth")
        .arg("--retrieved-docs")
        .arg(&docs)
        .assert()
        .failure()
        .stderr(contains("duplicate retrieved_docs id"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn mcp_import_autodetects_common_trace_shape() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_mcp_import_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("run.json");

    let mut cmd = Command::cargo_bin("raglens").unwrap();
    cmd.arg("mcp-import")
        .arg("--in")
        .arg(fixture_path("mcp/trace_basic.json"))
        .arg("--out")
        .arg(&out)
        .arg("--model")
        .arg("gpt-4.1")
        .arg("--top-k")
        .arg("5")
        .assert()
        .success();

    let written: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(written["question"], "Why did revenue increase?");
    assert_eq!(
        written["answer"],
        "Revenue increased due to US Direct growth."
    );
    assert_eq!(written["retrieved_docs"][0]["id"], "sales_us_direct");
    assert_eq!(written["context"]["model"], "gpt-4.1");
    assert_eq!(written["context"]["top_k"], 5);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn mcp_import_supports_custom_json_pointers() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("raglens_mcp_import_ptr_{stamp}"));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("run.json");

    let mut cmd = Command::cargo_bin("raglens").unwrap();
    cmd.arg("mcp-import")
        .arg("--in")
        .arg(fixture_path("mcp/trace_custom.json"))
        .arg("--out")
        .arg(&out)
        .arg("--question-pointer")
        .arg("/payload/q")
        .arg("--answer-pointer")
        .arg("/payload/final")
        .arg("--docs-pointer")
        .arg("/payload/ctx/hits")
        .assert()
        .success();

    let written: Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(written["question"], "How can I reduce unwanted robocalls?");
    assert_eq!(written["retrieved_docs"][0]["id"], "ftc_robocalls");
    assert_eq!(written["retrieved_docs"][0]["score"], 0.9);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn simulate_requires_queries_flag() {
    let mut cmd = Command::cargo_bin("rag-audit").unwrap();
    cmd.arg("simulate")
        .arg("examples/docs")
        .assert()
        .failure()
        .stderr(contains("--queries <QUERIES>"));
}
