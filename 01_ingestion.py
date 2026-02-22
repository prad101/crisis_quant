# Databricks notebook source
# MAGIC ## Notebook 01: Data Ingestion & EDA
# MAGIC **Purpose**: Load all raw datasets, handle HXL headers, coerce types, write to Delta tables.

# COMMAND ----------
from pyspark.sql import SparkSession
import pandas as pd
import os

spark = SparkSession.builder \
    .appName("humanitarian-ingestion") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sql("CREATE DATABASE IF NOT EXISTS humanitarian")
print("✓ Spark ready | Database 'humanitarian' ensured")

# COMMAND ----------
# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR  = "/dbfs/FileStore/humanitarian"   # adjust to your DBFS path
DATA_DIR  = "data"

# COMMAND ----------
# ── HELPERS ───────────────────────────────────────────────────────────────────

def safe_coerce(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Cast columns to numeric where ≥threshold fraction of values parse cleanly."""
    # Fix duplicate column names
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    ID_COLS = {
        "country_code", "countrycode", "iso3", "iso", "cluster_code",
        "clustercode", "adm1_code", "adm2_code", "adm3_code",
        "chfprojectcode", "externalprojectcode", "pooledFundId",
        "chfid", "donorcode",
    }

    for col in df.columns:
        if col.lower() in ID_COLS:
            continue
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().sum() / max(len(converted), 1) >= threshold:
                df[col] = converted
    return df


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV respecting HXL structure:
      Row 0 = human-readable headers
      Row 1 = HXL tags (starts with #)  → use as column names after cleaning
      Row 2+ = data
    Falls back to standard header if row 1 is not HXL.
    """
    df = pd.read_csv(file_path, header=None, dtype=str, encoding="utf-8-sig")

    if str(df.iloc[1].values[0]).startswith("#"):
        cols = [
            str(c).strip().replace("#", "").replace("+", "_").replace(" ", "_")
            for c in df.iloc[1].tolist()
        ]
        df = df.iloc[2:].reset_index(drop=True)
    else:
        cols = [
            str(c).strip().replace(" ", "_").replace("#", "").replace("+", "_")
            for c in df.iloc[0].tolist()
        ]
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = cols
    return df

# COMMAND ----------
# ── 1. STANDARD CSV DATASETS ──────────────────────────────────────────────────

datasets_to_load = {
    "fts_requirements": "fts_requirements_funding_globalcluster_global.csv",
    "population":       "cod_population_admin0.csv",
}

raw_datasets  = {}
df_data_dict  = {}

for name, fname in datasets_to_load.items():
    fpath = f"{ROOT_DIR}/{DATA_DIR}/{fname}"
    try:
        df = safe_coerce(load_csv(fpath))
        df_data_dict[name] = df
        sdf = spark.createDataFrame(df)
        raw_datasets[name] = sdf
        sdf.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(f"humanitarian.raw_{name}")
        print(f"✓ {name}: {sdf.count():,} rows → humanitarian.raw_{name}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"✗ {name}: {e}")

# COMMAND ----------
# ── 2. HNO 2025 (HXL format) ─────────────────────────────────────────────────

hno_path = f"{ROOT_DIR}/{DATA_DIR}/hpc_hno_2025.csv"
try:
    df_hno = load_csv(hno_path)
    df_hno["year"] = 2025
    df_hno = safe_coerce(df_hno)
    df_data_dict["hno"] = df_hno
    sdf_hno = spark.createDataFrame(df_hno)
    raw_datasets["hno"] = sdf_hno
    sdf_hno.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("humanitarian.raw_hno")
    print(f"✓ hno: {sdf_hno.count():,} rows → humanitarian.raw_hno")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"✗ hno: {e}")

# COMMAND ----------
# ── 3. PROJECT LEVEL DATA (no HXL) ───────────────────────────────────────────

project_files = {
    "projects":      "Projectleveldata/ProjectSummaryWithLocationAndCluster20260222055839817.csv",
    "contributions": "Projectleveldata/Contribution_by_Pooled_Fund_Code.csv",
}

for name, fname in project_files.items():
    fpath = f"{ROOT_DIR}/{DATA_DIR}/{fname}"
    try:
        df = pd.read_csv(fpath, dtype=str, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df = safe_coerce(df)
        df_data_dict[name] = df
        sdf = spark.createDataFrame(df)
        raw_datasets[name] = sdf
        sdf.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(f"humanitarian.raw_{name}")
        print(f"✓ {name}: {sdf.count():,} rows → humanitarian.raw_{name}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"✗ {name}: {e}")

# COMMAND ----------
# ── 4. EMDAT DISASTER DATA (Excel) ────────────────────────────────────────────

emdat_path = f"{ROOT_DIR}/{DATA_DIR}/public_emdat_incl_hist_2026-02-21.xlsx"
try:
    df_emdat = pd.read_excel(emdat_path, sheet_name="EM-DAT Data", dtype=str)
    df_emdat.columns = [
        c.strip().replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace("/", "_")
        for c in df_emdat.columns
    ]
    df_emdat = safe_coerce(df_emdat)
    df_data_dict["emdat"] = df_emdat
    sdf_emdat = spark.createDataFrame(df_emdat)
    raw_datasets["emdat"] = sdf_emdat
    sdf_emdat.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("humanitarian.raw_emdat")
    print(f"✓ emdat: {sdf_emdat.count():,} rows → humanitarian.raw_emdat")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"✗ emdat: {e}")

# COMMAND ----------
# ── 5. EDA SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)
for name, sdf in raw_datasets.items():
    print(f"\n{'─'*40}")
    print(f"  {name.upper()}: {sdf.count():,} rows × {len(sdf.columns)} cols")
    sdf.printSchema()

print(f"\n✓ Notebook 01 complete — {len(raw_datasets)} tables written to humanitarian.*")
