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
    cmd.arg("--fail-on-weak")
        .arg("0")
        .arg("readiness")
        .arg("examples/docs")
        .arg("--queries")
        .arg("examples/queries_structured.txt")
        .assert()
        .code(2);
}
