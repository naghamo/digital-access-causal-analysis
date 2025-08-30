# Impact of Home Computer Access on Student academic Achievement

*Causal inference analysis using PISA 2018*

> This repository contains the code and notebooks for our final project in **Introduction to Causal Inference (2024–2025)**. We estimate the causal effect of **home computer access** on students’ academic achievement using PISA data, and we run extensive **robustness checks** (trimming, imputation strategies, matching variants, random seeds, temporal stability, and placebos).

---

## 🔍 Research question

**Does having access to a computer at home influence a student’s academic achievement?**
Outcome = standardized PISA composite (average of math, reading, science).
Treatment = indicator for **computer access at home**.

---

## 🧭 Methods (brief)

* **Preprocessing:** MICE + mean/mode imputation; binary/ordinal + one-hot encoding.
* **Propensity score:** model selected by **Brier score**.
* **Overlap:** trimming (main: `[0.05, 0.95]`), caliper matching.
* **Outcome models:** Linear / Ridge / Tree-based; best by **MSE**.
* **Estimators:** **T-Learner**, **IPW**, **Doubly Robust**, **Matching (ATT)**.
* **Uncertainty:** **95% bootstrap CIs**.

---

## 📁 Repository structure

```
DIGITAL-ACCESS-CAUSAL-ANALYSIS/
├── data/                         # PISA data (compressed or prepared files)
├── Robustness_checks/
│   ├── Ambiguous_Covariates/
│   │   └── estimation_no_home_env.ipynb
│   ├── Covariate_Adjustment/
│   │   └── estimation_Covariate_Adjustment.ipynb
│   ├── Imputation_Robustness/
│   │   ├── data_loader_Complete_case.py
│   │   ├── estimation_imputation_complete_case.ipynb
│   │   └── estimation_imputation_simple.ipynb
│   ├── Matching_Method/
│   │   ├── estimation_matching_knn_1.ipynb
│   │   ├── estimation_matching_knn_3.ipynb
│   │   └── estimation_matching_knn_5.ipynb
│   ├── Overlap_Sensitivity_(Trimming)/
│   │   ├── estimation_no_trimming.ipynb
│   │   ├── estimation_trimming_0.1.ipynb
│   │   └── estimation_trimming_0.01.ipynb
│   ├── Placebo/
│   │   ├── estimation_Placebo_T_and_Y.ipynb
│   │   ├── estimation_Placebo_T.ipynb
│   │   └── estimation_Placebo_Y.ipynb
│   ├── random_seeds/
│   │   ├── estimation_seed=0.ipynb
│   │   └── estimation_seed=10.ipynb
│   ├── subgroups_analysis/
│   │   ├── estimation_edu_father_high.ipynb
│   │   ├── estimation_edu_father_low.ipynb
│   │   ├── estimation_edu_gender_F.ipynb
│   │   ├── estimation_edu_gender_M.ipynb
│   │   ├── estimation_edu_mother_high.ipynb
│   │   ├── estimation_edu_mother_low.ipynb
│   │   ├── estimation_SCE_high.ipynb
│   │   └── estimation_SCE_low.ipynb
│   └── Temporal_Stability_(Years)/
│       ├── estimation_2009.ipynb
│       ├── estimation_2012.ipynb
│       └── estimation_2015.ipynb
├── estimation.py                 # Main script to run the core analysis
├── data_loader.py                # Helpers for loading/cleaning PISA data
├── plot_robustness.ipynb         # Aggregated robustness plots
├── requirements.txt
└── README.md
```

---

## 📦 Data

* The PISA 2018 dataset is included (compressed) with the submission; for public use, obtain data from the **Learning Tower** repository or the **OECD PISA** site (see links in the appendix).
* Place the files under `data/` or update paths in `data_loader.py`.

---

## ▶️ Reproduce the main results

```bash
python estimation.py
```

Outputs:

* ATE/ATT estimates for **T-Learner, IPW, DR, Matching**
* 95% **bootstrap** confidence intervals
* Diagnostic plots (propensity overlap, calibration, covariate balance)


---

## ♻️ Reproduce robustness checks

Each check has its own notebook under `Robustness_checks/`:

* **Trimming / Overlap sensitivity:** `Overlap_Sensitivity_(Trimming)/`
* **Matching variants (KNN, caliper):** `Matching_Method/`
* **Imputation strategies:** `Imputation_Robustness/`
* **Covariate adjustment on/off:** `Covariate_Adjustment/`
* **Ambiguous covariates removed:** `Ambiguous_Covariates/`
* **Placebos (randomized T/Y):** `Placebo/`
* **Random seeds:** `random_seeds/`
* **Subgroups (SES, gender, parents’ education):** `subgroups_analysis/`
* **Temporal stability (2009/2012/2015):** `Temporal_Stability_(Years)/`

Use `plot_robustness.ipynb` to aggregate/visualize results across checks.

---

## 📊 Key outputs (expected)

* Consistently **positive, significant** effects across estimators (largest stability for **DR**).
* Diagnostics: good **propensity calibration**, **balance** after matching/weighting, and **null** placebo effects.

---

## 🔒 Reproducibility notes

* Main analyses use `random_seed = 42`; robustness notebooks vary seeds (0, 10, 42).
* Trimming default: `[0.05, 0.95]`; notebooks include alternatives.

---

## 📫 Contact

**Nagham Omar** & **Mahmoud Jabarin**
`{mahmoud.j, nagham.omar}@campus.technion.ac.il`

---

