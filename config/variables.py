"""
Machine-readable variable spec derived from reports/variable_triage.md.

Single source of truth for which columns play which causal role, so the
cleaning and WQS-modelling code never re-litigates the triage. Import from here;
do not hard-code column names elsewhere.
"""

# Identifier (linkage only; never a predictor)
ID = "ID_Identifier_base"

# --- Outcome ---------------------------------------------------------------
# Primary multivariate outcome vector (4 CHC domains; Gc/Know_Gc absent).
OUTCOMES = ["seq_gsm", "Simul_Gv", "Learn_Glr", "Plan_Gf"]
# Secondary composite -- report separately, keep OUT of the joint vector
# (it correlates 0.62-0.81 with the four domains, i.e. it IS their composite).
OUTCOME_COMPOSITE = "FCI_score"

# --- Exposure (WQS components, quantiled with a shared simplex weight) ------
EXPOSURES = ["PM25_prenatal", "PM25_age1", "PM2.5_age4"]

# --- Confounders (adjustment set) ------------------------------------------
# Core, parsimonious. *_RECODE / *_IMPUTE need preprocessing (see below).
CONFOUNDERS = ["M_age", "Child_sex", "ethnicity", "marital_status", "medlev"]
CONFOUNDERS_OPTIONAL = ["bplace"]  # only if SES signal is weak

# --- Effect modifier (ONE, pre-specified: the environmental-justice term) ---
MODIFIER_PRIMARY = "medlev"          # education / SES
MODIFIER_FALLBACK = "ses_index"      # constructed if medlev imputation unstable
MODIFIER_SENSITIVITY = "Child_sex"   # literature-standard, sensitivity only

# --- Mediators (NEVER in the adjustment set; reserve for mediation analysis) -
MEDIATORS = [
    "birthwt_age1", "BMI_age1", "gageanc", "b1placbir21",
    "Ht_Age1", "ht_age4", "Aual_age4", "Aheadcap_age4",
]

# --- Excluded entirely ------------------------------------------------------
EXCLUDE = [
    "smokecur",          # constant "No"
    "numbaby",           # only 2 twins -> near-zero variance
    "Occupation",        # 22.5% missing + redundant SES proxy
    "Ht_age7",           # measured AFTER the age-4 outcome (temporally invalid)
    "Ht_age4",           # duplicate height-at-4 (20 missing)
    "ht_age4.1",         # exact duplicate (Excel artifact)
    "Early_life_PM25",   # unconfirmed definition; redundant with the 3 windows
    "seq_gsm.1", "Simul_Gv.1", "Learn_Glr.1", "Plan_Gf.1",  # Excel duplicates
]

# --- Preprocessing directives (fit on TRAIN only, apply to VAL/TEST) --------
# Collapse high-cardinality / sparse categoricals.
COLLAPSE = {
    "marital_status": {  # strip whitespace first
        "Married": "Married", "Cohab": "Cohab",
        "Single": "Other", "Widowed": "Other", "Divorced": "Other",
    },
    # ethnicity: 14 messy/overlapping labels -> ~5 groups; resolve duplicates
    # (e.g. 'Frafra/Kusasi' folds into 'Dagarti/Grushi/Frafra/Kusasi',
    #  'Basare' into 'Konkomba/Basare', 'Chokosi' into 'Bimoba/Chokosi') and
    # bucket rare groups (<15) into 'Other'. Final mapping set in cleaning step.
}

# Error / sentinel codes to treat as missing before imputation.
MISSING_CODES = {"gageanc": [99]}

# Implausible ranges to flag/winsorize (data-entry errors).
IMPLAUSIBLE = {
    "ht_age4": (70, 130),       # cm for a 4-year-old
    "Aual_age4": (10, 40),
    "Aheadcap_age4": (40, 55),
}

# Columns needing imputation (Bayesian / model-based, fit on TRAIN only).
IMPUTE = ["medlev"]  # 43.5% missing
