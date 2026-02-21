-- =============================================================================
-- feature_engineering.sql
-- Medication Adherence Risk Model — Feature Engineering
--
-- PURPOSE:
--   Demonstrates how PDC, MPR, and refill gap features would be computed
--   in a production SQL environment (e.g., PostgreSQL, Snowflake, BigQuery).
--   The same logic is implemented in Python in src/feature_engineering.py
--   for local pipeline execution without a database.
--
-- ASSUMPTIONS:
--   - Table: refills (patient_id, rx_id, drug_class, fill_date, days_supply,
--                      fill_number, refill_gap_days, overlap_days)
--   - Table: patients (patient_id, age, chronic_flag)
--   - Observation window: 2022-01-01 to 2023-12-31
--   - PDC/MPR window: last 180 days of observation
--   - Non-adherence threshold: PDC < 0.80 in final 90-day window
-- =============================================================================


-- =============================================================================
-- 1. PROPORTION OF DAYS COVERED (PDC) — per patient, per 180-day window
--
--    PDC = unique covered days / window_days
--    Overlapping fills are NOT double-counted (DATE_SERIES trick below).
-- =============================================================================

WITH date_series AS (
    -- Generate one row per day in the observation window
    -- (PostgreSQL syntax; adjust generate_series for other dialects)
    SELECT
        patient_id,
        fill_date::date + gs AS coverage_day
    FROM refills
    CROSS JOIN generate_series(0, days_supply - 1) AS gs
    WHERE fill_date >= CURRENT_DATE - INTERVAL '180 days'
),
unique_covered_days AS (
    SELECT
        patient_id,
        COUNT(DISTINCT coverage_day) AS days_covered
    FROM date_series
    GROUP BY patient_id
),
pdc_calc AS (
    SELECT
        patient_id,
        days_covered,
        180 AS window_days,
        ROUND(days_covered::numeric / 180, 4) AS prior_pdc
    FROM unique_covered_days
)
SELECT * FROM pdc_calc;


-- =============================================================================
-- 2. MEDICATION POSSESSION RATIO (MPR) — per patient, per 180-day window
--
--    MPR = total days supply dispensed / window_days
--    Unlike PDC, this CAN exceed 1.0 (stockpiling). We cap at 1.0.
-- =============================================================================

WITH mpr_calc AS (
    SELECT
        patient_id,
        SUM(days_supply)                    AS total_days_supply,
        180                                 AS window_days,
        LEAST(
            ROUND(SUM(days_supply)::numeric / 180, 4),
            1.0
        )                                   AS prior_mpr
    FROM refills
    WHERE fill_date >= CURRENT_DATE - INTERVAL '180 days'
    GROUP BY patient_id
)
SELECT * FROM mpr_calc;


-- =============================================================================
-- 3. REFILL GAP FEATURES — rolling statistics per patient
--
--    Captures how often and how severely a patient delays refills.
-- =============================================================================

WITH gap_features AS (
    SELECT
        patient_id,
        AVG(refill_gap_days)                        AS avg_refill_gap_days,
        STDDEV(refill_gap_days)                     AS refill_gap_std,
        MAX(refill_gap_days)                        AS max_refill_gap_days,
        SUM(overlap_days)                           AS total_overlap_days,
        COUNT(refill_id)                            AS num_fills,
        -- Rolling 90-day gap trend (are gaps getting worse?)
        AVG(refill_gap_days) FILTER (
            WHERE fill_date >= CURRENT_DATE - INTERVAL '90 days'
        )                                           AS recent_avg_gap_90d,
        AVG(refill_gap_days) FILTER (
            WHERE fill_date BETWEEN
                CURRENT_DATE - INTERVAL '180 days'
                AND CURRENT_DATE - INTERVAL '90 days'
        )                                           AS prior_avg_gap_90d
    FROM refills
    GROUP BY patient_id
),
gap_trend AS (
    SELECT
        *,
        -- Positive = gap is getting worse over time
        COALESCE(recent_avg_gap_90d - prior_avg_gap_90d, 0) AS gap_trend_delta
    FROM gap_features
)
SELECT * FROM gap_trend;


-- =============================================================================
-- 4. PRESCRIPTION-LEVEL FEATURES — medication complexity
-- =============================================================================

SELECT
    patient_id,
    COUNT(DISTINCT rx_id)                           AS num_prescriptions,
    COUNT(DISTINCT drug_class)                      AS num_drug_classes,
    -- Medication switches: distinct classes minus 1 baseline
    GREATEST(COUNT(DISTINCT drug_class) - 1, 0)    AS switch_count,
    ROUND(AVG(days_supply), 1)                      AS avg_days_supply,
    -- Proportion of fills that had any overlap (early refill)
    ROUND(
        SUM(CASE WHEN overlap_days > 0 THEN 1 ELSE 0 END)::numeric
        / NULLIF(COUNT(*), 0),
        4
    )                                               AS pct_fills_with_overlap
FROM refills
GROUP BY patient_id;


-- =============================================================================
-- 5. LABEL GENERATION — non_adherent_90d
--
--    A patient is labeled non-adherent if their PDC in the final 90-day
--    window drops below 0.80 — the standard CMS/NCQA adherence threshold.
-- =============================================================================

WITH final_window_coverage AS (
    SELECT
        patient_id,
        fill_date::date + gs AS coverage_day
    FROM refills
    CROSS JOIN generate_series(0, days_supply - 1) AS gs
    WHERE fill_date >= CURRENT_DATE - INTERVAL '90 days'
),
covered AS (
    SELECT
        patient_id,
        COUNT(DISTINCT coverage_day) AS days_covered_90d
    FROM final_window_coverage
    GROUP BY patient_id
),
all_patients AS (
    SELECT patient_id FROM patients
),
label AS (
    SELECT
        p.patient_id,
        COALESCE(c.days_covered_90d, 0)         AS days_covered_90d,
        ROUND(
            COALESCE(c.days_covered_90d, 0)::numeric / 90,
            4
        )                                         AS pdc_90d,
        CASE
            WHEN COALESCE(c.days_covered_90d, 0)::numeric / 90 < 0.80
            THEN 1 ELSE 0
        END                                       AS non_adherent_90d
    FROM all_patients p
    LEFT JOIN covered c USING (patient_id)
)
SELECT * FROM label;


-- =============================================================================
-- 6. FINAL FEATURE TABLE — joins all of the above
-- =============================================================================

WITH
pdc           AS ( /* ... paste PDC CTE above ... */ SELECT NULL::text AS patient_id, NULL::numeric AS prior_pdc ),
mpr           AS ( /* ... paste MPR CTE above ... */ SELECT NULL::text AS patient_id, NULL::numeric AS prior_mpr ),
gaps          AS ( /* ... paste gap CTE above  ... */ SELECT NULL::text AS patient_id, NULL::numeric AS avg_refill_gap_days, NULL::numeric AS refill_gap_std, NULL::numeric AS max_refill_gap_days, NULL::int AS total_overlap_days ),
rx_feats      AS ( /* ... paste rx  CTE above  ... */ SELECT NULL::text AS patient_id, NULL::int AS num_prescriptions, NULL::int AS num_drug_classes, NULL::int AS switch_count, NULL::numeric AS avg_days_supply ),
labels        AS ( /* ... paste label CTE above... */ SELECT NULL::text AS patient_id, NULL::int AS non_adherent_90d )
SELECT
    pt.patient_id,
    pt.age,
    pt.chronic_flag,
    COALESCE(rx.num_prescriptions, 0)   AS num_prescriptions,
    COALESCE(rx.avg_days_supply, 30)    AS avg_days_supply,
    COALESCE(rx.num_drug_classes, 0)    AS num_drug_classes,
    COALESCE(rx.switch_count, 0)        AS switch_count,
    COALESCE(pdc.prior_pdc, 0)          AS prior_pdc,
    COALESCE(mpr.prior_mpr, 0)          AS prior_mpr,
    COALESCE(g.avg_refill_gap_days, 0)  AS avg_refill_gap_days,
    COALESCE(g.refill_gap_std, 0)       AS refill_gap_std,
    COALESCE(g.max_refill_gap_days, 0)  AS max_refill_gap_days,
    COALESCE(g.total_overlap_days, 0)   AS overlap_days,
    COALESCE(lb.non_adherent_90d, 1)    AS non_adherent_90d   -- default non-adherent if no fills
FROM patients pt
LEFT JOIN pdc    USING (patient_id)
LEFT JOIN mpr    USING (patient_id)
LEFT JOIN gaps g USING (patient_id)
LEFT JOIN rx_feats rx USING (patient_id)
LEFT JOIN labels lb   USING (patient_id);
