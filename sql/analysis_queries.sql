-- Reporting queries (portable: work on MySQL and SQLite).
-- Each block starts with "-- name: <title>"; src/sql_loader.py runs them after loading.

-- name: KPI summary
SELECT COUNT(*)                                   AS total_candidates,
       ROUND(AVG(accepted) * 100, 2)              AS placement_rate_pct,
       ROUND(AVG(interview_score), 2)             AS avg_interview_score,
       ROUND(AVG(skills_match_percentage), 2)     AS avg_skills_match_pct,
       ROUND(AVG(CASE WHEN risk_band = 'High Risk' THEN 1.0 ELSE 0.0 END) * 100, 2) AS high_risk_pct
FROM job_placement_cleaned;

-- name: Acceptance rate by company tier
SELECT company_tier, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS acceptance_rate_pct
FROM job_placement_cleaned
GROUP BY company_tier
ORDER BY company_tier;

-- name: Placement rate by competition level
SELECT competition_level, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY competition_level
ORDER BY placement_rate_pct DESC;

-- name: Experience vs placement success
SELECT experience_category, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY experience_category
ORDER BY placement_rate_pct DESC;

-- name: Skills match level vs placement
SELECT skills_match_level, COUNT(*) AS candidates,
       ROUND(AVG(interview_score), 2) AS avg_interview_score,
       ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY skills_match_level
ORDER BY placement_rate_pct DESC;

-- name: Certification impact on acceptance
SELECT certifications_count, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY certifications_count
ORDER BY certifications_count;

-- name: Bond requirement and job-role match effect
SELECT bond_requirement, job_role_match, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY bond_requirement, job_role_match
ORDER BY bond_requirement, job_role_match;

-- name: Specialization-wise placement
SELECT degree_specialization, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS placement_rate_pct
FROM job_placement_cleaned
GROUP BY degree_specialization
ORDER BY placement_rate_pct DESC;

-- name: Risk band distribution (model output)
SELECT risk_band, COUNT(*) AS candidates, ROUND(AVG(accepted) * 100, 2) AS actual_placement_rate_pct,
       ROUND(AVG(placement_probability_score), 3) AS avg_predicted_probability
FROM job_placement_cleaned
GROUP BY risk_band
ORDER BY avg_predicted_probability;

-- name: Candidates to prioritise (eligible, high probability not yet placed)
SELECT candidate_id, degree_specialization, company_tier, technical_score, skills_match_percentage,
       placement_probability_score
FROM job_placement_cleaned
WHERE accepted = 0 AND placement_probability_score >= 0.7
ORDER BY placement_probability_score DESC
LIMIT 20;
