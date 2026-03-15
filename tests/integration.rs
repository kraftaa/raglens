use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;

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
