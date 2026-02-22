# Humanitarian Aid Intelligence — Full Pipeline

## Architecture

```
data/
  hpc_hno_2025.csv                          ← HXL format, needs+beneficiaries
  fts_requirements_funding_*.csv            ← HXL format, funding requirements
  cod_population_admin0.csv                 ← Population reference
  Projectleveldata/
    ProjectSummaryWithLocationAndCluster*.csv ← CBPF project-level (geo + cluster)
    Contribution_by_Pooled_Fund_Code.csv    ← Donor contributions
  public_emdat_incl_hist_*.xlsx             ← Disaster context

Databricks Notebooks (run in order):
  01_ingestion.py    → Load CSVs/Excel → humanitarian.raw_* Delta tables
  02_pipeline.py     → Join + feature engineering → humanitarian.features
  03_anomaly.py      → IsolationForest + KMeans + MLflow → humanitarian.anomalies

Streamlit App:
  app.py             → Interactive dashboard (reads Delta or uses synthetic data)
```

## Datasets Selected & Why

| Dataset | Why |
|---|---|
| `hpc_hno_2025` | People in need + targeted — core beneficiary signal |
| `fts_requirements` | Requirements vs funded per country+cluster+year |
| `ProjectSummaryWithLocationAndCluster` | Project-level geo + org + cluster + budget |
| `Contribution_by_Pooled_Fund_Code` | Donor flows — pledge vs paid |
| `cod_population_admin0` | Per-capita normalization |
| `public_emdat` | Disaster severity context for anomaly scoring |

**Excluded**: `fts_incoming/outgoing/internal` (overlap with requirements), `HRP` (redundant), `PipelineProject*` (subset of ProjectSummary)

## Features Engineered

### Funding Efficiency
- `funding_coverage_rate` — fraction of requirements actually funded
- `funding_gap_usd` / `funding_gap_pct` — absolute and % shortfall
- `funding_per_capita` / `requirement_per_capita` — per-person normalization

### Beneficiary Impact
- `beneficiary_to_funding_ratio` — people targeted per dollar funded
- `need_to_requirements_ratio` — people in need per dollar requested
- `targeting_coverage_rate` — % of people in need being targeted
- `cost_per_beneficiary` — USD per person targeted

### Context
- `disaster_severity_score` — composite score from EMDAT (deaths + affected + frequency)
- `sector_funding_efficiency_pct` — sector-wide funding rate
- `project_vs_sector_avg` — how this project compares to its sector

## ML Models

### Isolation Forest (Anomaly Detection)
- Features: 9 engineered metrics
- Contamination: 5%
- Output: `is_anomaly`, `anomaly_score` per country-cluster-year row
- MLflow tracked: contamination, anomaly_count, anomaly_rate

### K-Means (Benchmarking)
- Auto-selects K (4–12) by silhouette score
- Assigns each project to a "comparable peer group"
- MLflow tracked: K, inertia, silhouette_score

### Portfolio Optimization (scipy.linprog)
- Maximize: Σ(beneficiaries_i × allocation_i)
- Subject to: Σ(requirements_i × allocation_i) ≤ BUDGET
- Run interactively in Streamlit with budget slider

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Databricks Setup

1. Upload all data files to DBFS: `/FileStore/humanitarian/data/`
2. Run `01_ingestion.py` → creates `humanitarian.raw_*` tables
3. Run `02_pipeline.py` → creates `humanitarian.features`, `.projects`, `.contributions`
4. Run `03_anomaly.py` → creates `humanitarian.anomalies`, logs to MLflow
5. Deploy `app.py` as a Databricks App or export and run via Streamlit Cloud
