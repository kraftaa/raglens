use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct AnswerAuditDriver {
    pub label: String,
    pub baseline: f64,
    pub current: f64,
    pub delta: f64,
    pub contribution_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerAuditIssue {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MentionAssessment {
    pub segment: String,
    pub claimed_direction: String,
    pub actual_direction: String,
    pub delta: f64,
    pub contribution_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerAuditReport {
    pub verdict: String,
    pub question: Option<String>,
    pub answer: String,
    pub baseline_period: String,
    pub current_period: String,
    pub total_delta: f64,
    pub top_drivers: Vec<AnswerAuditDriver>,
    pub mentions: Vec<MentionAssessment>,
    pub issues: Vec<AnswerAuditIssue>,
    pub coverage_pct: f64,
    pub alignment_score: f64,
    pub confidence: String,
}

#[derive(Debug, Clone)]
pub struct AnswerAuditRequest {
    pub data: std::path::PathBuf,
    period_granularity: PeriodBucket,
    pub group_by: Vec<String>,
    pub metric: String,
    pub period_col: String,
    pub baseline: String,
    pub current: String,
    pub question: Option<String>,
    pub answer: String,
    pub weak_contribution_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct AnswerAuditInput {
    pub data: std::path::PathBuf,
    pub auto: bool,
    pub period_granularity: String,
    pub group_by: Vec<String>,
    pub metric: Option<String>,
    pub period_col: Option<String>,
    pub baseline: Option<String>,
    pub current: Option<String>,
    pub question: Option<String>,
    pub answer: String,
    pub weak_contribution_threshold: f64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Direction {
    Increase,
    Decrease,
    Flat,
    Unknown,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PeriodBucket {
    Raw,
    Month,
    Week,
}

impl Direction {
    fn as_str(self) -> &'static str {
        match self {
            Direction::Increase => "increase",
            Direction::Decrease => "decrease",
            Direction::Flat => "flat",
            Direction::Unknown => "unknown",
        }
    }
}

impl PeriodBucket {
    fn as_str(self) -> &'static str {
        match self {
            PeriodBucket::Raw => "raw",
            PeriodBucket::Month => "month",
            PeriodBucket::Week => "week",
        }
    }
}

#[derive(Debug, Clone)]
struct InternalDriver {
    key_values: Vec<String>,
    baseline: f64,
    current: f64,
    delta: f64,
    contribution_pct: f64,
}

#[derive(Debug, Clone)]
struct SegmentStat {
    segment: String,
    delta: f64,
    contribution_pct: f64,
}

#[derive(Debug, Clone)]
struct ColumnProfile {
    name: String,
    numeric_ratio: f64,
    unique_values: Vec<String>,
}

#[derive(Debug, Clone)]
struct CsvProfile {
    headers: Vec<String>,
    columns: Vec<ColumnProfile>,
}

pub fn resolve_input(input: AnswerAuditInput) -> Result<(AnswerAuditRequest, Vec<String>)> {
    if input.weak_contribution_threshold < 0.0 {
        anyhow::bail!("weak_contribution_threshold must be >= 0");
    }
    let granularity = parse_period_granularity(&input.period_granularity)?;
    let profile = profile_csv(&input.data)?;
    let mut notes = Vec::new();

    let metric = match input.metric {
        Some(m) => {
            ensure_has_column(&profile, &m)?;
            m
        }
        None if input.auto => {
            let inferred = infer_metric_column(&profile)?;
            notes.push(format!("auto metric: {}", inferred));
            inferred
        }
        None => anyhow::bail!("missing --metric (or use --auto)"),
    };

    let period_col = match input.period_col {
        Some(p) => {
            ensure_has_column(&profile, &p)?;
            p
        }
        None if input.auto => {
            let inferred = infer_period_column(&profile, &metric)?;
            notes.push(format!("auto period_col: {}", inferred));
            inferred
        }
        None => anyhow::bail!("missing --period-col (or use --auto)"),
    };

    let period_values = bucket_unique_values(
        unique_values_for_column(&profile, &period_col)?,
        granularity,
    );
    if period_values.len() < 2 {
        if granularity == PeriodBucket::Raw {
            anyhow::bail!(
                "period column '{}' must contain at least two values",
                period_col
            );
        }
        anyhow::bail!(
            "period column '{}' must contain at least two parseable values for '{}' granularity",
            period_col,
            granularity.as_str()
        );
    }
    let (auto_baseline, auto_current) = infer_period_pair(&period_values)?;

    let baseline = match input.baseline {
        Some(b) => b,
        None if input.auto => {
            let inferred = auto_baseline.clone();
            notes.push(format!("auto baseline: {}", inferred));
            inferred
        }
        None => anyhow::bail!("missing --baseline (or use --auto)"),
    };

    let current = match input.current {
        Some(c) => c,
        None if input.auto => {
            let inferred = if auto_current != baseline {
                auto_current.clone()
            } else {
                period_values
                    .iter()
                    .rev()
                    .find(|v| **v != baseline)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("cannot infer current period"))?
            };
            notes.push(format!("auto current: {}", inferred));
            inferred
        }
        None => anyhow::bail!("missing --current (or use --auto)"),
    };

    let baseline = normalize_user_period_value("baseline", &baseline, granularity)?;
    let current = normalize_user_period_value("current", &current, granularity)?;
    if input.auto || granularity != PeriodBucket::Raw {
        notes.push(format!("period granularity: {}", granularity.as_str()));
    }

    if baseline == current {
        anyhow::bail!("baseline and current must be different");
    }

    if !period_values.contains(&baseline) {
        anyhow::bail!(
            "baseline '{}' not found in period column '{}'",
            baseline,
            period_col
        );
    }
    if !period_values.contains(&current) {
        anyhow::bail!(
            "current '{}' not found in period column '{}'",
            current,
            period_col
        );
    }

    let group_by = if input.group_by.is_empty() {
        if input.auto {
            let inferred = infer_group_by_columns(&profile, &metric, &period_col);
            if inferred.is_empty() {
                anyhow::bail!("could not infer group-by columns; pass --group-by explicitly");
            }
            notes.push(format!("auto group_by: {}", inferred.join(",")));
            inferred
        } else {
            anyhow::bail!("answer-audit requires --group-by (or use --auto)")
        }
    } else {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for col in input.group_by {
            ensure_has_column(&profile, &col)?;
            if seen.insert(col.clone()) {
                out.push(col);
            }
        }
        out
    };

    Ok((
        AnswerAuditRequest {
            data: input.data,
            period_granularity: granularity,
            group_by,
            metric,
            period_col,
            baseline,
            current,
            question: input.question,
            answer: input.answer,
            weak_contribution_threshold: input.weak_contribution_threshold,
        },
        notes,
    ))
}

pub fn audit_answer(req: &AnswerAuditRequest) -> Result<AnswerAuditReport> {
    if req.group_by.is_empty() {
        anyhow::bail!("answer-audit requires at least one --group-by column");
    }
    if req.weak_contribution_threshold < 0.0 {
        anyhow::bail!("weak_contribution_threshold must be >= 0");
    }
    let (drivers, total_delta, skipped_period_rows) = load_and_compute(
        &req.data,
        &req.group_by,
        &req.metric,
        &req.period_col,
        req.period_granularity,
        &req.baseline,
        &req.current,
    )?;

    if drivers.is_empty() {
        anyhow::bail!(
            "no rows found for baseline '{}' and current '{}'",
            req.baseline,
            req.current
        );
    }

    let segment_stats = aggregate_segments(&drivers, total_delta);
    let mut mentions = Vec::new();
    let mut issues = Vec::new();

    let claimed_global = infer_global_claim_direction(&req.answer);
    let mut contradiction_count = 0usize;
    let mut misattribution_count = 0usize;

    for segment in &segment_stats {
        if !mentions_value(&req.answer, &segment.segment) {
            continue;
        }
        let actual = direction_from_delta(segment.delta);
        let claimed = claimed_global;
        mentions.push(MentionAssessment {
            segment: segment.segment.clone(),
            claimed_direction: claimed.as_str().to_string(),
            actual_direction: actual.as_str().to_string(),
            delta: segment.delta,
            contribution_pct: segment.contribution_pct,
        });

        if (claimed == Direction::Increase && actual == Direction::Decrease)
            || (claimed == Direction::Decrease && actual == Direction::Increase)
        {
            contradiction_count += 1;
            issues.push(AnswerAuditIssue {
                code: "CONTRADICTION".to_string(),
                message: format!(
                    "{} is {} ({:+.2}), but answer implies {}",
                    segment.segment,
                    actual.as_str(),
                    segment.delta,
                    claimed.as_str()
                ),
            });
        }

        let weak_pct = req.weak_contribution_threshold * 100.0;
        if segment.contribution_pct.abs() < weak_pct {
            misattribution_count += 1;
            issues.push(AnswerAuditIssue {
                code: "WEAK_SIGNAL".to_string(),
                message: format!(
                    "{} contribution is weak ({:+.1}%)",
                    segment.segment, segment.contribution_pct
                ),
            });
        }
    }

    if let Some(top_driver) = top_primary_driver(&drivers, total_delta) {
        let mentioned = top_driver
            .key_values
            .iter()
            .any(|v| mentions_value(&req.answer, v));
        if !mentioned {
            issues.push(AnswerAuditIssue {
                code: "MISSING_DRIVER".to_string(),
                message: format!(
                    "Main driver {} not mentioned ({:+.2}, {:+.1}%)",
                    top_driver.key_values.join(" | "),
                    top_driver.delta,
                    top_driver.contribution_pct
                ),
            });
        }
    }

    if mentions.is_empty() {
        issues.push(AnswerAuditIssue {
            code: "NO_DATA_CLAIMS".to_string(),
            message: "Answer does not mention recognizable data segments".to_string(),
        });
    }
    if skipped_period_rows > 0 {
        issues.push(AnswerAuditIssue {
            code: "DATA_QUALITY".to_string(),
            message: format!(
                "{} rows were skipped due to unparseable period values for '{}' granularity",
                skipped_period_rows,
                req.period_granularity.as_str()
            ),
        });
    }

    let top_n = drivers.iter().take(3).collect::<Vec<_>>();
    let covered = top_n
        .iter()
        .filter(|d| d.key_values.iter().any(|v| mentions_value(&req.answer, v)))
        .count();
    let coverage_pct = if top_n.is_empty() {
        0.0
    } else {
        covered as f64 / top_n.len() as f64 * 100.0
    };

    let missing_major = issues.iter().any(|i| i.code == "MISSING_DRIVER");
    let verdict = if contradiction_count > 0 || missing_major {
        "INCORRECT"
    } else if !issues.is_empty() {
        "RISKY"
    } else {
        "SUPPORTED"
    }
    .to_string();

    let mut score = 1.0f64;
    score -= 0.4 * contradiction_count as f64;
    score -= 0.3 * usize::from(missing_major) as f64;
    score -= 0.1 * misattribution_count as f64;
    score = score.clamp(0.0, 1.0);

    let confidence = if contradiction_count > 0 || missing_major {
        "HIGH"
    } else if !issues.is_empty() {
        "MEDIUM"
    } else {
        "LOW"
    }
    .to_string();

    let top_drivers = drivers
        .iter()
        .take(5)
        .map(|d| AnswerAuditDriver {
            label: d.key_values.join(" | "),
            baseline: d.baseline,
            current: d.current,
            delta: d.delta,
            contribution_pct: d.contribution_pct,
        })
        .collect::<Vec<_>>();

    Ok(AnswerAuditReport {
        verdict,
        question: req.question.clone(),
        answer: req.answer.clone(),
        baseline_period: req.baseline.clone(),
        current_period: req.current.clone(),
        total_delta,
        top_drivers,
        mentions,
        issues,
        coverage_pct,
        alignment_score: score,
        confidence,
    })
}

fn profile_csv(data: &Path) -> Result<CsvProfile> {
    let mut rdr = csv::Reader::from_path(data)
        .with_context(|| format!("opening CSV data {}", data.display()))?;
    let headers = rdr
        .headers()
        .with_context(|| format!("reading headers from {}", data.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect::<Vec<_>>();
    if headers.is_empty() {
        anyhow::bail!("CSV has no headers");
    }
    let mut uniques = vec![HashSet::<String>::new(); headers.len()];
    let mut non_empty = vec![0usize; headers.len()];
    let mut numeric = vec![0usize; headers.len()];

    for rec in rdr.records() {
        let row = rec.with_context(|| format!("reading record from {}", data.display()))?;
        for i in 0..headers.len() {
            let value = row.get(i).unwrap_or("").trim();
            if value.is_empty() {
                continue;
            }
            non_empty[i] += 1;
            uniques[i].insert(value.to_string());
            if value.parse::<f64>().is_ok() {
                numeric[i] += 1;
            }
        }
    }

    let columns = headers
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let mut unique_values = uniques[i].iter().cloned().collect::<Vec<_>>();
            unique_values.sort();
            let ratio = if non_empty[i] == 0 {
                0.0
            } else {
                numeric[i] as f64 / non_empty[i] as f64
            };
            ColumnProfile {
                name: name.clone(),
                numeric_ratio: ratio,
                unique_values,
            }
        })
        .collect::<Vec<_>>();

    Ok(CsvProfile { headers, columns })
}

fn ensure_has_column(profile: &CsvProfile, name: &str) -> Result<()> {
    if profile.headers.iter().any(|h| h == name) {
        Ok(())
    } else {
        anyhow::bail!("column '{}' not found in CSV", name)
    }
}

fn infer_metric_column(profile: &CsvProfile) -> Result<String> {
    let mut candidates = profile
        .columns
        .iter()
        .filter(|c| c.numeric_ratio >= 0.9 && !is_period_name(&c.name))
        .map(|c| (metric_name_score(&c.name), c.name.clone()))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    candidates
        .first()
        .map(|(_, name)| name.clone())
        .ok_or_else(|| anyhow::anyhow!("could not infer numeric metric column"))
}

fn infer_period_column(profile: &CsvProfile, metric: &str) -> Result<String> {
    if let Some(col) = profile
        .columns
        .iter()
        .find(|c| c.name != metric && is_period_name(&c.name))
    {
        return Ok(col.name.clone());
    }
    profile
        .columns
        .iter()
        .find(|c| c.name != metric && c.unique_values.len() >= 2 && c.unique_values.len() <= 32)
        .map(|c| c.name.clone())
        .ok_or_else(|| anyhow::anyhow!("could not infer period column"))
}

fn infer_group_by_columns(profile: &CsvProfile, metric: &str, period_col: &str) -> Vec<String> {
    profile
        .columns
        .iter()
        .filter(|c| c.name != metric && c.name != period_col)
        .filter(|c| c.unique_values.len() >= 2 && c.unique_values.len() <= 100)
        .take(2)
        .map(|c| c.name.clone())
        .collect::<Vec<_>>()
}

fn unique_values_for_column(profile: &CsvProfile, name: &str) -> Result<Vec<String>> {
    profile
        .columns
        .iter()
        .find(|c| c.name == name)
        .map(|c| c.unique_values.clone())
        .ok_or_else(|| anyhow::anyhow!("column '{}' not found", name))
}

fn is_period_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["period", "date", "time", "month", "quarter", "year", "week"]
        .iter()
        .any(|k| n.contains(k))
}

fn metric_name_score(name: &str) -> i32 {
    let n = name.to_ascii_lowercase();
    let mut score = 0i32;
    for k in ["revenue", "sales", "gmv", "arr", "mrr", "amount", "value"] {
        if n.contains(k) {
            score += 100;
        }
    }
    for k in ["usd", "eur", "gbp", "jpy", "money", "price"] {
        if n.contains(k) {
            score += 40;
        }
    }
    for k in ["cost", "expense", "spend"] {
        if n.contains(k) {
            score += 25;
        }
    }
    for k in ["orders", "units", "qty"] {
        if n.contains(k) {
            score += 10;
        }
    }
    for k in ["rate", "ratio", "pct", "percent"] {
        if n.contains(k) {
            score -= 50;
        }
    }
    for k in ["traffic", "session", "visit", "click", "impression"] {
        if n.contains(k) {
            score -= 30;
        }
    }
    score
}

fn parse_period_granularity(value: &str) -> Result<PeriodBucket> {
    match value.trim().to_ascii_lowercase().as_str() {
        "raw" => Ok(PeriodBucket::Raw),
        "month" => Ok(PeriodBucket::Month),
        "week" => Ok(PeriodBucket::Week),
        other => anyhow::bail!("unsupported period granularity '{}'", other),
    }
}

fn bucket_unique_values(values: Vec<String>, granularity: PeriodBucket) -> Vec<String> {
    let mut set = HashSet::new();
    for value in values {
        let bucketed = bucket_period_value(&value, granularity);
        if !bucketed.is_empty() {
            set.insert(bucketed);
        }
    }
    let mut out = set.into_iter().collect::<Vec<_>>();
    out.sort();
    out
}

fn infer_period_pair(values: &[String]) -> Result<(String, String)> {
    if values.len() < 2 {
        anyhow::bail!("cannot infer period pair from fewer than 2 unique values");
    }
    let lookup = values
        .iter()
        .map(|v| (v.to_ascii_lowercase(), v.clone()))
        .collect::<HashMap<_, _>>();

    let aliases = [
        ("old", "new"),
        ("previous", "current"),
        ("prior", "current"),
        ("before", "after"),
        ("baseline", "current"),
        ("past", "present"),
        ("t0", "t1"),
    ];
    for (b, c) in aliases {
        if let (Some(base), Some(curr)) = (lookup.get(b), lookup.get(c)) {
            return Ok((base.clone(), curr.clone()));
        }
    }

    // Numeric periods: choose min -> max
    let numeric = values
        .iter()
        .map(|v| v.parse::<f64>().ok().map(|n| (n, v.clone())))
        .collect::<Option<Vec<_>>>();
    if let Some(mut nums) = numeric {
        nums.sort_by(|a, b| a.0.total_cmp(&b.0));
        return Ok((
            nums.first()
                .map(|x| x.1.clone())
                .ok_or_else(|| anyhow::anyhow!("cannot infer baseline period"))?,
            nums.last()
                .map(|x| x.1.clone())
                .ok_or_else(|| anyhow::anyhow!("cannot infer current period"))?,
        ));
    }

    // Fallback deterministic lexical order.
    let mut sorted = values.to_vec();
    sorted.sort();
    Ok((
        sorted
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("cannot infer baseline period"))?,
        sorted
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("cannot infer current period"))?,
    ))
}

fn bucket_period_value(raw: &str, granularity: PeriodBucket) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match granularity {
        PeriodBucket::Raw => trimmed.to_string(),
        PeriodBucket::Month => parse_period_date(trimmed)
            .map(|d| format!("{:04}-{:02}", d.year(), d.month()))
            .unwrap_or_default(),
        PeriodBucket::Week => parse_period_date(trimmed)
            .map(|d| {
                let iso = d.iso_week();
                format!("{:04}-W{:02}", iso.year(), iso.week())
            })
            .unwrap_or_default(),
    }
}

fn normalize_user_period_value(
    field: &str,
    value: &str,
    granularity: PeriodBucket,
) -> Result<String> {
    let normalized = bucket_period_value(value, granularity);
    if granularity != PeriodBucket::Raw && value.trim().is_empty() {
        anyhow::bail!("{} period value cannot be empty", field);
    }
    if granularity != PeriodBucket::Raw && normalized.is_empty() {
        anyhow::bail!(
            "{} period '{}' is not parseable for '{}' granularity",
            field,
            value,
            granularity.as_str()
        );
    }
    Ok(normalized)
}

fn parse_period_date(input: &str) -> Option<NaiveDate> {
    if looks_like_year_month(input, '-') {
        return NaiveDate::parse_from_str(&format!("{input}-01"), "%Y-%m-%d").ok();
    }
    if looks_like_year_month(input, '/') {
        return NaiveDate::parse_from_str(&format!("{input}/01"), "%Y/%m/%d").ok();
    }

    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"] {
        if let Ok(d) = NaiveDate::parse_from_str(input, fmt) {
            return Some(d);
        }
    }

    let normalized = input.strip_suffix('Z').unwrap_or(input);
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
    ] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(normalized, fmt) {
            return Some(dt.date());
        }
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
        return Some(dt.date_naive());
    }
    None
}

fn looks_like_year_month(input: &str, sep: char) -> bool {
    let mut parts = input.split(sep);
    let (Some(y), Some(m), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    y.len() == 4
        && m.len() == 2
        && y.chars().all(|c| c.is_ascii_digit())
        && m.chars().all(|c| c.is_ascii_digit())
}

fn load_and_compute(
    data: &Path,
    group_by: &[String],
    metric: &str,
    period_col: &str,
    period_granularity: PeriodBucket,
    baseline: &str,
    current: &str,
) -> Result<(Vec<InternalDriver>, f64, usize)> {
    let mut rdr = csv::Reader::from_path(data)
        .with_context(|| format!("opening CSV data {}", data.display()))?;
    let headers = rdr
        .headers()
        .with_context(|| format!("reading headers from {}", data.display()))?
        .clone();

    let mut idx = HashMap::<String, usize>::new();
    for (i, name) in headers.iter().enumerate() {
        idx.insert(name.trim().to_string(), i);
    }

    let metric_idx = *idx
        .get(metric)
        .ok_or_else(|| anyhow::anyhow!("metric column '{}' not found", metric))?;
    let period_idx = *idx
        .get(period_col)
        .ok_or_else(|| anyhow::anyhow!("period column '{}' not found", period_col))?;
    let group_idxs = group_by
        .iter()
        .map(|c| {
            idx.get(c)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("group-by column '{}' not found", c))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut base_map = HashMap::<String, f64>::new();
    let mut curr_map = HashMap::<String, f64>::new();
    let mut key_values_map = HashMap::<String, Vec<String>>::new();
    let mut skipped_period_rows = 0usize;

    for rec in rdr.records() {
        let row = rec.with_context(|| format!("reading record from {}", data.display()))?;
        let raw_period = row.get(period_idx).unwrap_or("");
        let period = bucket_period_value(raw_period, period_granularity);
        if period_granularity != PeriodBucket::Raw
            && !raw_period.trim().is_empty()
            && period.is_empty()
        {
            skipped_period_rows += 1;
            continue;
        }
        if period != baseline && period != current {
            continue;
        }
        let metric_value = parse_metric_value(row.get(metric_idx).unwrap_or(""))?;
        let values = group_idxs
            .iter()
            .map(|i| row.get(*i).unwrap_or("").trim().to_string())
            .collect::<Vec<_>>();
        let key = values.join("\u{1f}");
        key_values_map.entry(key.clone()).or_insert(values);
        if period == baseline {
            *base_map.entry(key).or_insert(0.0) += metric_value;
        } else {
            *curr_map.entry(key).or_insert(0.0) += metric_value;
        }
    }

    let mut keys = key_values_map.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    let mut drivers = Vec::new();
    let mut total_delta = 0.0f64;
    for key in &keys {
        let baseline_v = *base_map.get(key).unwrap_or(&0.0);
        let current_v = *curr_map.get(key).unwrap_or(&0.0);
        let delta = current_v - baseline_v;
        total_delta += delta;
        drivers.push(InternalDriver {
            key_values: key_values_map.get(key).cloned().unwrap_or_default(),
            baseline: baseline_v,
            current: current_v,
            delta,
            contribution_pct: 0.0,
        });
    }
    if total_delta.abs() < f64::EPSILON {
        for d in &mut drivers {
            d.contribution_pct = 0.0;
        }
    } else {
        for d in &mut drivers {
            d.contribution_pct = d.delta / total_delta * 100.0;
        }
    }
    drivers.sort_by(|a, b| b.delta.abs().total_cmp(&a.delta.abs()));
    Ok((drivers, total_delta, skipped_period_rows))
}

fn aggregate_segments(drivers: &[InternalDriver], total_delta: f64) -> Vec<SegmentStat> {
    let mut map = HashMap::<String, f64>::new();
    for d in drivers {
        for value in &d.key_values {
            *map.entry(value.clone()).or_insert(0.0) += d.delta;
        }
    }
    let mut out = map
        .into_iter()
        .map(|(segment, delta)| SegmentStat {
            segment,
            delta,
            contribution_pct: if total_delta.abs() < f64::EPSILON {
                0.0
            } else {
                delta / total_delta * 100.0
            },
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| {
        b.contribution_pct
            .abs()
            .total_cmp(&a.contribution_pct.abs())
            .then_with(|| a.segment.cmp(&b.segment))
    });
    out
}

fn top_primary_driver(drivers: &[InternalDriver], total_delta: f64) -> Option<&InternalDriver> {
    if drivers.is_empty() {
        return None;
    }
    if total_delta > 0.0 {
        drivers
            .iter()
            .filter(|d| d.delta > 0.0)
            .max_by(|a, b| a.contribution_pct.total_cmp(&b.contribution_pct))
            .or_else(|| drivers.first())
    } else if total_delta < 0.0 {
        drivers
            .iter()
            .filter(|d| d.delta < 0.0)
            .max_by(|a, b| {
                a.contribution_pct
                    .abs()
                    .total_cmp(&b.contribution_pct.abs())
            })
            .or_else(|| drivers.first())
    } else {
        drivers.first()
    }
}

fn mentions_value(answer: &str, value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }

    // Keep very short alpha tokens strict/exact-token to avoid false positives ("us" in "business").
    if value.len() <= 2 && value.chars().all(|c| c.is_ascii_alphabetic()) {
        let value_lc = value.to_ascii_lowercase();
        return normalized_tokens(answer)
            .into_iter()
            .any(|tok| tok == value_lc);
    }

    let answer_norm = normalized_tokens(answer);
    let value_norm = normalized_tokens(value);
    if value_norm.is_empty() {
        return false;
    }

    let haystack = format!(" {} ", answer_norm.join(" "));
    let phrase = value_norm.join(" ");
    let phrase_needle = format!(" {} ", phrase);
    if haystack.contains(&phrase_needle) {
        return true;
    }

    value_norm
        .iter()
        .all(|tok| haystack.contains(&format!(" {} ", tok)))
}

fn tokenize_preserve_case(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in input.chars() {
        if ch.is_alphanumeric() {
            buf.push(ch);
        } else if !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn normalized_tokens(input: &str) -> Vec<String> {
    tokenize_preserve_case(input)
        .into_iter()
        .map(|t| t.chars().flat_map(|c| c.to_lowercase()).collect::<String>())
        .collect()
}

fn infer_global_claim_direction(answer: &str) -> Direction {
    let tokens = normalized_tokens(answer);
    let inc = has_any_token(
        &tokens,
        &[
            "increase",
            "increased",
            "growth",
            "grew",
            "up",
            "rise",
            "rising",
            "higher",
        ],
    );
    let dec = has_any_token(
        &tokens,
        &[
            "decrease",
            "decreased",
            "decline",
            "declined",
            "down",
            "drop",
            "dropped",
            "lower",
            "fell",
            "falling",
        ],
    );

    match (inc, dec) {
        (true, false) => Direction::Increase,
        (false, true) => Direction::Decrease,
        _ => Direction::Unknown,
    }
}

fn has_any_token(tokens: &[String], words: &[&str]) -> bool {
    let set = tokens.iter().map(String::as_str).collect::<HashSet<_>>();
    words.iter().any(|w| set.contains(w))
}

fn parse_metric_value(raw: &str) -> Result<f64> {
    let trimmed = raw.trim();
    let mut normalized = trimmed.replace(',', "");
    normalized = normalized
        .trim_start_matches(['$', '€', '£', '¥'])
        .to_string();
    let is_wrapped_negative = normalized.starts_with('(') && normalized.ends_with(')');
    if is_wrapped_negative {
        normalized = normalized
            .trim_start_matches('(')
            .trim_end_matches(')')
            .to_string();
    }
    let value = normalized
        .parse::<f64>()
        .with_context(|| format!("invalid numeric metric '{}'", raw))?;
    Ok(if is_wrapped_negative { -value } else { value })
}

fn direction_from_delta(delta: f64) -> Direction {
    if delta > 0.0 {
        Direction::Increase
    } else if delta < 0.0 {
        Direction::Decrease
    } else {
        Direction::Flat
    }
}

#[cfg(test)]
mod tests {
    use super::{
        audit_answer, metric_name_score, resolve_input, AnswerAuditInput, AnswerAuditRequest,
        PeriodBucket,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn flags_contradiction_and_missing_driver() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("raglens_answer_audit_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,100\nUS,Direct,new,140\nEU,Partner,old,200\nEU,Partner,new,180\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Raw,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "period".to_string(),
            baseline: "old".to_string(),
            current: "new".to_string(),
            question: Some("Why did revenue increase?".to_string()),
            answer: "Revenue increased due to EU growth".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert_eq!(out.verdict, "INCORRECT");
        assert!(out.issues.iter().any(|i| i.code == "CONTRADICTION"));
        assert!(out.issues.iter().any(|i| i.code == "MISSING_DRIVER"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn supported_when_main_driver_matches_answer() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("raglens_answer_audit_supported_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,100\nUS,Direct,new,150\nEU,Partner,old,120\nEU,Partner,new,125\nAPAC,Online,old,80\nAPAC,Online,new,78\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Raw,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "period".to_string(),
            baseline: "old".to_string(),
            current: "new".to_string(),
            question: None,
            answer: "Revenue increased due to strong US Direct growth".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert_eq!(out.verdict, "SUPPORTED");
        assert!(out.issues.is_empty());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn risky_when_answer_mentions_weak_driver() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("raglens_answer_audit_risky_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,100\nUS,Direct,new,160\nLATAM,Online,old,10\nLATAM,Online,new,11\nEU,Partner,old,120\nEU,Partner,new,118\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Raw,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "period".to_string(),
            baseline: "old".to_string(),
            current: "new".to_string(),
            question: None,
            answer: "Revenue increased due to US Direct and LATAM growth".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert_eq!(out.verdict, "RISKY");
        assert!(out.issues.iter().any(|i| i.code == "WEAK_SIGNAL"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn metric_name_score_prefers_revenue_over_traffic() {
        assert!(metric_name_score("revenue_usd") > metric_name_score("traffic"));
        assert!(metric_name_score("sales_amount") > metric_name_score("conversion_rate"));
    }

    #[test]
    fn month_granularity_rejects_non_date_period_values() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("raglens_answer_audit_bad_period_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,100\nUS,Direct,new,120\n",
        )
        .expect("write csv");

        let input = AnswerAuditInput {
            data: path.clone(),
            auto: false,
            period_granularity: "month".to_string(),
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: Some("revenue".to_string()),
            period_col: Some("period".to_string()),
            baseline: Some("old".to_string()),
            current: Some("new".to_string()),
            question: None,
            answer: "Revenue increased due to US growth".to_string(),
            weak_contribution_threshold: 0.1,
        };

        let err = resolve_input(input).expect_err("should reject non-date periods for month");
        assert!(err.to_string().contains("at least two parseable values"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn ignores_substring_false_positive_for_direction_claim() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raglens_answer_audit_direction_false_pos_{stamp}.csv"
        ));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,100\nUS,Direct,new,140\nEU,Partner,old,200\nEU,Partner,new,180\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Raw,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "period".to_string(),
            baseline: "old".to_string(),
            current: "new".to_string(),
            question: None,
            answer: "The setup was reviewed; EU policy changed".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert!(!out.issues.iter().any(|i| i.code == "CONTRADICTION"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_thousands_separated_metric_values() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("raglens_answer_audit_numeric_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,period,revenue\nUS,Direct,old,\"1,000\"\nUS,Direct,new,\"1,200\"\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Raw,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "period".to_string(),
            baseline: "old".to_string(),
            current: "new".to_string(),
            question: None,
            answer: "US Direct increased".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert_eq!(out.total_delta, 200.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn month_granularity_accepts_rfc3339_offset_dates() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("raglens_answer_audit_rfc3339_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,date,revenue\nUS,Direct,2026-01-03T10:00:00+00:00,100\nUS,Direct,2026-02-03T10:00:00+00:00,120\nEU,Partner,2026-01-04T08:30:00+00:00,90\nEU,Partner,2026-02-05T08:30:00+00:00,95\n",
        )
        .expect("write csv");

        let input = AnswerAuditInput {
            data: path.clone(),
            auto: true,
            period_granularity: "month".to_string(),
            group_by: vec![],
            metric: None,
            period_col: None,
            baseline: None,
            current: None,
            question: None,
            answer: "US grew".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let (resolved, _) = resolve_input(input).expect("auto resolve should work");
        assert_eq!(resolved.baseline, "2026-01");
        assert_eq!(resolved.current, "2026-02");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reports_data_quality_when_some_period_rows_are_unparseable() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("raglens_answer_audit_data_quality_{stamp}.csv"));
        fs::write(
            &path,
            "region,channel,date,revenue\nUS,Direct,2026-01-03,100\nUS,Direct,2026-02-03,120\nEU,Partner,not-a-date,15\n",
        )
        .expect("write csv");

        let req = AnswerAuditRequest {
            data: path.clone(),
            period_granularity: PeriodBucket::Month,
            group_by: vec!["region".to_string(), "channel".to_string()],
            metric: "revenue".to_string(),
            period_col: "date".to_string(),
            baseline: "2026-01".to_string(),
            current: "2026-02".to_string(),
            question: None,
            answer: "US increased".to_string(),
            weak_contribution_threshold: 0.1,
        };
        let out = audit_answer(&req).expect("audit should work");
        assert!(out.issues.iter().any(|i| i.code == "DATA_QUALITY"));

        let _ = fs::remove_file(path);
    }
}
