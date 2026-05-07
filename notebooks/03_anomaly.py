# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook 03: Anomaly Detection + Benchmarking (MLflow tracked)
# MAGIC **Purpose**: Run Isolation Forest anomaly detection and K-Means benchmarking on features.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.0 --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, pandas_udf
from pyspark.sql.types import DoubleType, IntegerType, StringType
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

spark = SparkSession.builder \
    .appName("humanitarian-anomaly") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sql("CREATE DATABASE IF NOT EXISTS humanitarian")
print("✓ Ready")

# COMMAND ----------

# ── LOAD FEATURES ─────────────────────────────────────────────────────────────
print("="*70)
print("LOADING FEATURES")
print("="*70)

features_df = spark.table("humanitarian.features")
print(f"✓ Loaded: {features_df.count():,} rows × {len(features_df.columns)} cols")

# COMMAND ----------

display(features_df.columns)

# COMMAND ----------

# ── FEATURE SELECTION FOR ML ──────────────────────────────────────────────────
ML_FEATURES = [
    "funding_coverage_rate",
    "funding_gap_pct",
    "beneficiary_to_funding_ratio",
    "need_to_requirements_ratio",
    "targeting_coverage_rate",
    "cost_per_beneficiary",
    "disaster_severity_score",
    "sector_funding_efficiency_pct",
    "project_vs_sector_avg",
]

# Filter to rows with valid ML features and convert to pandas
# df_ml = features_df.select(
#     ["country_code", "cluster_name", "year", "appeal_name",
#      "total_requirements_usd", "total_funding_usd"] + ML_FEATURES
# ).dropna(subset=ML_FEATURES).toPandas()

df_ml = features_df.select(
    ["country_code", "cluster_name", "year", "total_requirements_usd", "total_funding_usd"] + ML_FEATURES
).dropna(subset=ML_FEATURES).toPandas()

print(f"✓ ML-ready rows (no nulls in features): {len(df_ml):,}")

# COMMAND ----------

# ── ISOLATION FOREST ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ISOLATION FOREST ANOMALY DETECTION")
print("="*70)

CONTAMINATION = 0.05
N_ESTIMATORS  = 100

X = df_ml[ML_FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with mlflow.start_run(run_name="isolation_forest_v1"):
    iso = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=42
    )
    iso.fit(X_scaled)
    predictions  = iso.predict(X_scaled)    # 1 = normal, -1 = anomaly
    scores = iso.decision_function(X_scaled)  # higher = more normal

    df_ml["is_anomaly"] = (predictions == -1).astype(int)
    df_ml["anomaly_score"] = -scores   # flip: higher = more anomalous

    anomaly_count = int(df_ml["is_anomaly"].sum())
    anomaly_rate  = float(df_ml["is_anomaly"].mean())

    mlflow.log_param("contamination", CONTAMINATION)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("n_features", len(ML_FEATURES))
    mlflow.log_metric("anomaly_count", anomaly_count)
    mlflow.log_metric("anomaly_rate", anomaly_rate)
    mlflow.sklearn.log_model(iso, "isolation_forest_model")
    mlflow.sklearn.log_model(scaler, "scaler")

    print(f"✓ Anomalies detected: {anomaly_count:,} ({anomaly_rate*100:.1f}%)")

# COMMAND ----------

# ── K-MEANS BENCHMARKING ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("K-MEANS BENCHMARKING (Comparable Project Clusters)")
print("="*70)

K = 8
best_k   = K
best_sil = -1

with mlflow.start_run(run_name="kmeans_benchmarking_v1"):
    # Try K from 4 to 12 and pick best silhouette
    for k in range(4, 13):
        km_trial = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels   = km_trial.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
            if sil > best_sil:
                best_sil = sil
                best_k   = k

    print(f"  Best K={best_k} (silhouette={best_sil:.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_ml["cluster_id"] = kmeans.fit_predict(X_scaled)

    mlflow.log_param("n_clusters", best_k)
    mlflow.log_metric("inertia", kmeans.inertia_)
    mlflow.log_metric("silhouette_score", best_sil)
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    print(f"✓ K-Means done: {best_k} clusters, inertia={kmeans.inertia_:.0f}")
    print(f"  Cluster distribution:\n{pd.Series(df_ml['cluster_id']).value_counts().sort_index()}")

# COMMAND ----------

# ── WRITE ANOMALIES DELTA TABLE ───────────────────────────────────────────────
print("\n" + "="*70)
print("WRITING humanitarian.anomalies")
print("="*70)

sdf_anomalies = spark.createDataFrame(df_ml)
sdf_anomalies.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("humanitarian.anomalies")

print(f"✓ humanitarian.anomalies: {sdf_anomalies.count():,} rows")

# COMMAND ----------

# ── TOP ANOMALIES SUMMARY ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 20 ANOMALOUS ALLOCATIONS")
print("="*70)

top_anomalies = df_ml[df_ml["is_anomaly"] == 1] \
    .sort_values("anomaly_score", ascending=False) \
    .head(20)[["country_code", "cluster_name", "year",
               "total_requirements_usd", "total_funding_usd",
               "funding_gap_pct", "anomaly_score", "cluster_id"]]

# top_anomalies = df_ml[df_ml["is_anomaly"] == 1] \
#     .sort_values("anomaly_score", ascending=False) \
#     .head(20)[["country_code", "cluster_name", "year", "appeal_name",
#                "total_requirements_usd", "total_funding_usd",
#                "funding_gap_pct", "anomaly_score", "cluster_id"]]

print(top_anomalies.to_string(index=False))
print("\n✓ Notebook 03 complete")

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS humanitarian")
spark.sql("CREATE SCHEMA IF NOT EXISTS humanitarian.data")
spark.sql("CREATE VOLUME IF NOT EXISTS humanitarian.data.files")

# COMMAND ----------


