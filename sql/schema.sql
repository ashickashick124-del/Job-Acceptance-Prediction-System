-- MySQL schema for the cleaned + engineered placement dataset.
-- You can run this file in MySQL Workbench / the VS Code SQLTools extension,
-- or just run:  python -m src.sql_loader   (it creates the database and table for you).

CREATE DATABASE IF NOT EXISTS hr_placement CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE hr_placement;

DROP TABLE IF EXISTS job_placement_cleaned;

CREATE TABLE job_placement_cleaned (
    candidate_id                INT UNSIGNED  NOT NULL,
    age_years                   TINYINT UNSIGNED NOT NULL,
    gender                      VARCHAR(10)   NOT NULL,
    ssc_percentage              DECIMAL(5,2)  NOT NULL,
    hsc_percentage              DECIMAL(5,2)  NOT NULL,
    degree_percentage           DECIMAL(5,2)  NOT NULL,
    degree_specialization       VARCHAR(30)   NOT NULL,
    technical_score             DECIMAL(5,2)  NOT NULL,
    aptitude_score              DECIMAL(5,2)  NOT NULL,
    communication_score         DECIMAL(5,2)  NOT NULL,
    skills_match_percentage     DECIMAL(5,2)  NOT NULL,
    certifications_count        TINYINT UNSIGNED NOT NULL,
    internship_experience       VARCHAR(3)    NOT NULL,
    years_of_experience         TINYINT UNSIGNED NOT NULL,
    career_switch_willingness   VARCHAR(12)   NOT NULL,
    relevant_experience         VARCHAR(12)   NOT NULL,
    previous_ctc_lpa            DECIMAL(6,2)  NOT NULL,
    expected_ctc_lpa            DECIMAL(6,2)  NOT NULL,
    company_tier                VARCHAR(6)    NOT NULL,
    job_role_match              VARCHAR(12)   NOT NULL,
    competition_level           VARCHAR(6)    NOT NULL,
    bond_requirement            VARCHAR(12)   NOT NULL,
    notice_period_days          SMALLINT UNSIGNED NOT NULL,
    layoff_history              VARCHAR(3)    NOT NULL,
    employment_gap_months       SMALLINT UNSIGNED NOT NULL,
    relocation_willingness      VARCHAR(12)   NOT NULL,
    status                      VARCHAR(10)   NOT NULL,
    accepted                    TINYINT(1)    NOT NULL,
    -- engineered features
    interview_score             DECIMAL(5,2)  NOT NULL,
    avg_academic_pct            DECIMAL(5,2)  NOT NULL,
    ctc_gap_lpa                 DECIMAL(7,2)  NOT NULL,
    ctc_ratio                   DECIMAL(8,3)  NOT NULL,
    experience_category         VARCHAR(10)   NOT NULL,
    academic_band               VARCHAR(10)   NOT NULL,
    skills_match_level          VARCHAR(6)    NOT NULL,
    interview_performance       VARCHAR(6)    NOT NULL,
    -- model output (out-of-fold predicted probability of being placed)
    placement_probability_score DECIMAL(6,4)  NULL,
    risk_band                   VARCHAR(12)   NULL,
    PRIMARY KEY (candidate_id),
    INDEX idx_status (status),
    INDEX idx_company_tier (company_tier),
    INDEX idx_specialization (degree_specialization),
    INDEX idx_risk_band (risk_band)
) ENGINE=InnoDB;
