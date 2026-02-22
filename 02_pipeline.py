# Databricks notebook source
# MAGIC ## Notebook 02: PySpark Pipeline & Feature Engineering
# MAGIC **Purpose**: Join all datasets, engineer features, write to humanitarian.features Delta table.

# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, lower, trim, sum as spark_sum,
    avg, count, max as spark_max, min as spark_min,
    coalesce, regexp_replace, to_date, year as spark_year,
    countDistinct, isnan, isnull
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("humanitarian-pipeline") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sql("CREATE DATABASE IF NOT EXISTS humanitarian")
print("✓ Spark ready")

# COMMAND ----------
# ── LOAD RAW DELTA TABLES ─────────────────────────────────────────────────────
print("="*70)
print("LOADING RAW DELTA TABLES")
print("="*70)

tables = ["hno", "fts_requirements", "population", "projects", "contributions", "emdat"]
dfs = {}
for t in tables:
    try:
        dfs[t] = spark.table(f"humanitarian.raw_{t}")
        print(f"✓ {t}: {dfs[t].count():,} rows")
    except Exception as e:
        print(f"✗ {t}: {e}")

# COMMAND ----------
# ── STEP 1: PREP FTS REQUIREMENTS ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: PREP FTS REQUIREMENTS (Base Funding Table)")
print("="*70)

fts = dfs["fts_requirements"].select(
    col("country_code").cast(StringType()).alias("country_code"),
    when(col("clusterCode").rlike(r"^\d+$"), col("clusterCode")).otherwise(None).alias("cluster_id"),
    lower(trim(col("cluster"))).alias("cluster_name"),
    when(col("year").rlike(r"^\d{4}$"), col("year").cast(IntegerType())).otherwise(None).alias("year"),
    col("name").alias("appeal_name"),
    col("id").alias("appeal_id"),
    when(col("requirements").rlike(r"^-?\d+(\.\d+)?$"),
         col("requirements").cast(DoubleType())).otherwise(None).alias("total_requirements_usd"),
    when(col("funding").rlike(r"^-?\d+(\.\d+)?$"),
         col("funding").cast(DoubleType())).otherwise(None).alias("total_funding_usd"),
    when(col("percentFunded").rlike(r"^-?\d+(\.\d+)?$"),
         col("percentFunded").cast(DoubleType())).otherwise(None).alias("funding_pct"),
).dropna(subset=["country_code", "year"])

fts = fts.fillna({"total_funding_usd": 0.0, "total_requirements_usd": 0.0})
print(f"✓ FTS prep: {fts.count():,} rows")

# COMMAND ----------
# ── STEP 2: PREP HNO ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: PREP HNO (Needs & Beneficiaries)")
print("="*70)

# Sector normalization map: HNO multilingual → canonical English
SECTOR_MAP = {
    # English
    "food security": "food security", "nutrition": "nutrition", "health": "health",
    "education": "education", "agriculture": "agriculture", "logistics": "logistics",
    "protection": "protection", "child protection": "protection - child protection",
    "protection - child protection": "protection - child protection",
    "gender-based violence (gbv)": "protection - gender-based violence",
    "protection - gender-based violence": "protection - gender-based violence",
    "mine action": "protection - mine action",
    "protection - mine action": "protection - mine action",
    "water, sanitation and hygiene": "wash", "water sanitation hygiene": "wash", "wash": "wash",
    "emergency shelter and nfi": "shelter and nfi", "shelter and nfi": "shelter and nfi",
    "shelter and non-food items": "shelter and nfi", "shelter and nfis": "shelter and nfi",
    "multipurpose cash": "multipurpose cash", "multi-purpose cash": "multipurpose cash",
    "emergency telecommunications": "emergency telecommunications",
    "camp coordination / management": "cccm",
    "camp coordination and camp management": "cccm", "cccm": "cccm",
    "housing, land and property": "protection - housing, land and property",
    "protection - housing, land and property": "protection - housing, land and property",
    "early recovery": "early recovery", "coordination": "coordination",
    "coordination and support services": "coordination", "refugee response": "refugee response",
    "general protection": "protection", "protection (overall)": "protection",
    # French
    "santé": "health", "éducation": "education", "logistique": "logistics",
    "sécurité alimentaire": "food security", "nutrition": "nutrition",
    "eau, hygiène et assainissement": "wash", "eau, hygiene et assainissement": "wash",
    "abris et biens non-alimentaires": "shelter and nfi", "abris": "shelter and nfi",
    "abris&nfi": "shelter and nfi",
    "protection de l'enfant": "protection - child protection",
    "violence basée sur le genre (vbg)": "protection - gender-based violence",
    "lutte antimines": "protection - mine action",
    "logement, terres et propriété": "protection - housing, land and property",
    "réponse aux réfugiés": "refugee response",
    "protection générale": "protection",
    "coordination et organisation de camp": "cccm",
    "cash à usage multiple caseload": "multipurpose cash",
    "tranferts monetaires a usage multiples": "multipurpose cash",
    "transfert en espèce à usage multiple": "multipurpose cash",
    "cash a usage multiple": "multipurpose cash",
    "articles ménagers essentiels": "shelter and nfi",
    "récupération rapide": "early recovery", "récupération précoce": "early recovery",
    # Spanish
    "salud": "health", "educación": "education", "educación en emergencias": "education",
    "seguridad alimentaria y nutrición": "food security", "seguridad alimentaria": "food security",
    "agua, saneamiento e higiene": "wash",
    "albergue y artículos no alimentarios": "shelter and nfi",
    "protección de la infancia": "protection - child protection",
    "violencia de género (vdg)": "protection - gender-based violence",
    "acción contra minas": "protection - mine action",
    "protección (total)": "protection", "protección": "protection",
    "protección general": "protection",
    "coordinación y gestión de campamentos": "cccm",
    "coordinación y gestión de albergues": "cccm",
    "recuperación temprana": "early recovery",
    "nutrición": "nutrition",
    # Variants with response type suffixes
    "food security and livelihoods": "food security",
    "food security conflict response": "food security",
    "food security natural disasters response": "food security",
    "health conflict response": "health",
    "water, sanitation and hygiene conflict response": "wash",
    "shelter/nfis conflict response": "shelter and nfi", "total - snfi": "shelter and nfi",
    "camp coordination and camp management conflict response": "cccm",
    "camp coordination and camp management natural disaster response": "cccm",
    "child protection natural disaster": "protection - child protection",
    "gender based violence natural disaster": "protection - gender-based violence",
    "general protection natural disaster": "protection",
    "education natural disasters response": "education",
    "anticipatory - food security": "food security", "readiness - food security": "food security",
    "anticipatory - wash": "wash", "readiness - wash": "wash", "total - wash": "wash",
    "anticipatory - nutrition": "nutrition", "total - nutrition": "nutrition",
    "anticipatory - education": "education", "readiness - education": "education",
    "total - education": "education",
    "anticipatory - cccm": "cccm", "readiness - cccm": "cccm", "total - cccm": "cccm",
    "anticipatory - snfi": "shelter and nfi",
    "anticipatory - protection (overall)": "protection",
    "readiness - early recovery": "early recovery", "total - early recovery": "early recovery",
    "mpc caseload": "multipurpose cash",
}

# Build Spark expression for normalization
def build_normalizer(input_col: str):
    from functools import reduce
    expr = lit(None).cast(StringType())
    for raw, canonical in SECTOR_MAP.items():
        expr = when(lower(trim(col(input_col))) == raw, lit(canonical)).otherwise(expr)
    return expr

hno = dfs["hno"].withColumn("sector_normalized", build_normalizer("sector_description"))

# Aggregate HNO to country level (year is 2025 for all)
hno_agg = hno.groupBy("country_code").agg(
    spark_sum(when(col("inneed").cast(DoubleType()).isNotNull(),
                   col("inneed").cast(DoubleType()))).alias("total_inneed"),
    spark_sum(when(col("targeted").cast(DoubleType()).isNotNull(),
                   col("targeted").cast(DoubleType()))).alias("total_targeted"),
    spark_sum(when(col("population").cast(DoubleType()).isNotNull(),
                   col("population").cast(DoubleType()))).alias("hno_population"),
)

print(f"✓ HNO aggregated: {hno_agg.count():,} country rows")

# COMMAND ----------
# ── STEP 3: PREP POPULATION ───────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: PREP POPULATION")
print("="*70)

pop = dfs["population"]
pop_agg = pop.select(
    col("ISO3").alias("country_code"),
    when(col("Population").cast(DoubleType()).isNotNull(),
         col("Population").cast(DoubleType())).otherwise(None).alias("population"),
    when(col("Reference_year").cast(IntegerType()).isNotNull(),
         col("Reference_year").cast(IntegerType())).otherwise(None).alias("ref_year"),
).groupBy("country_code").agg(
    spark_sum("population").alias("total_population")
)

print(f"✓ Population aggregated: {pop_agg.count():,} country rows")

# COMMAND ----------
# ── STEP 4: PREP PROJECTS (CBPF Project Level) ────────────────────────────────
print("\n" + "="*70)
print("STEP 4: PREP PROJECTS")
print("="*70)

proj = dfs["projects"]
proj_cols = proj.columns
print(f"  Available columns: {proj_cols}")

proj_prep = proj.select(
    col("PooledFundName").alias("country_name"),
    col("ChfProjectCode").alias("project_code"),
    col("OrganizationName").alias("org_name"),
    col("OrganizationType").alias("org_type"),
    col("ProjectTitle").alias("project_title"),
    col("AllocationYear").cast(IntegerType()).alias("year"),
    col("AllocationType").alias("allocation_type"),
    when(col("Budget").cast(DoubleType()).isNotNull(),
         col("Budget").cast(DoubleType())).otherwise(None).alias("budget"),
    col("ProjectStatus").alias("project_status"),
    col("Cluster").alias("cluster"),
    when(col("ClusterPercentage").cast(DoubleType()).isNotNull(),
         col("ClusterPercentage").cast(DoubleType())).otherwise(None).alias("cluster_pct"),
    col("AdminLocation1").alias("admin_location"),
    when(col("AdminLocation1Latitude").cast(DoubleType()).isNotNull(),
         col("AdminLocation1Latitude").cast(DoubleType())).otherwise(None).alias("latitude"),
    when(col("AdminLocation1Longitude").cast(DoubleType()).isNotNull(),
         col("AdminLocation1Longitude").cast(DoubleType())).otherwise(None).alias("longitude"),
    col("GenderMarker").alias("gender_marker"),
    col("EnvironmentMarker").alias("environment_marker"),
    col("ActualStartDate").alias("start_date"),
    col("ActualEndDate").alias("end_date"),
).dropna(subset=["project_code", "year"])

print(f"✓ Projects prep: {proj_prep.count():,} rows")

# COMMAND ----------
# ── STEP 5: PREP CONTRIBUTIONS (Donor Flows) ──────────────────────────────────
print("\n" + "="*70)
print("STEP 5: PREP CONTRIBUTIONS")
print("="*70)

contrib = dfs["contributions"]
contrib_prep = contrib.select(
    col("PooledFundName").alias("country_name"),
    col("PooledFundId").cast(IntegerType()).alias("fund_id"),
    col("FiscalYear").cast(IntegerType()).alias("year"),
    col("DonorName").alias("donor_name"),
    col("CountryCode").alias("donor_country_code"),
    when(col("PledgeAmt").cast(DoubleType()).isNotNull(),
         col("PledgeAmt").cast(DoubleType())).otherwise(None).alias("pledge_usd"),
    when(col("PaidAmt").cast(DoubleType()).isNotNull(),
         col("PaidAmt").cast(DoubleType())).otherwise(None).alias("paid_usd"),
).dropna(subset=["fund_id", "year"])

# Country-year donor aggregations
contrib_agg = contrib_prep.groupBy("country_name", "year").agg(
    spark_sum("pledge_usd").alias("total_pledged_usd"),
    spark_sum("paid_usd").alias("total_paid_usd"),
    countDistinct("donor_name").alias("num_donors"),
)

print(f"✓ Contributions aggregated: {contrib_agg.count():,} rows")

# COMMAND ----------
# ── STEP 6: PREP EMDAT (Disaster Context) ─────────────────────────────────────
print("\n" + "="*70)
print("STEP 6: PREP EMDAT DISASTER DATA")
print("="*70)

emdat = dfs["emdat"]
emdat_prep = emdat.select(
    col("ISO").alias("country_code"),
    col("Disaster_Type").alias("disaster_type"),
    col("Disaster_Subtype").alias("disaster_subtype"),
    when(col("Start_Year").cast(IntegerType()).isNotNull(),
         col("Start_Year").cast(IntegerType())).otherwise(None).alias("year"),
    when(col("Total_Deaths").cast(DoubleType()).isNotNull(),
         col("Total_Deaths").cast(DoubleType())).otherwise(None).alias("total_deaths"),
    when(col("Total_Affected").cast(DoubleType()).isNotNull(),
         col("Total_Affected").cast(DoubleType())).otherwise(None).alias("total_affected_disaster"),
    when(col("Total_Damage_000_US$").cast(DoubleType()).isNotNull(),
         col("Total_Damage_000_US$").cast(DoubleType())).otherwise(None).alias("total_damage_000usd"),
).dropna(subset=["country_code", "year"])

# Aggregate recent disasters (last 5 years relative to data)
emdat_agg = emdat_prep.filter(col("year") >= 2019).groupBy("country_code").agg(
    count("disaster_type").alias("num_disasters_5yr"),
    spark_sum("total_deaths").alias("total_disaster_deaths_5yr"),
    spark_sum("total_affected_disaster").alias("total_disaster_affected_5yr"),
    spark_sum("total_damage_000usd").alias("total_disaster_damage_000usd_5yr"),
)

print(f"✓ EMDAT aggregated: {emdat_agg.count():,} country rows")

# COMMAND ----------
# ── STEP 7: BUILD FEATURES TABLE ──────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 7: BUILDING FEATURES TABLE (FTS base + joins)")
print("="*70)

# Base: FTS Requirements (10K rows, country+year+cluster level)
features_df = fts
print(f"Base (FTS): {features_df.count():,} rows")

# Join HNO (country level — no year in HNO, it's all 2025)
features_df = features_df.join(hno_agg, on="country_code", how="left")
print(f"After HNO join: {features_df.count():,} rows")

# Join Population
features_df = features_df.join(pop_agg, on="country_code", how="left")
print(f"After Population join: {features_df.count():,} rows")

# Join EMDAT
features_df = features_df.join(emdat_agg, on="country_code", how="left")
print(f"After EMDAT join: {features_df.count():,} rows")

# Fill nulls
features_df = features_df.fillna({
    "total_funding_usd": 0.0,
    "total_requirements_usd": 0.0,
    "total_inneed": 0.0,
    "total_targeted": 0.0,
    "total_population": 0.0,
    "num_disasters_5yr": 0,
    "total_disaster_deaths_5yr": 0.0,
    "total_disaster_affected_5yr": 0.0,
})

# COMMAND ----------
# ── STEP 8: FEATURE ENGINEERING ───────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 8: FEATURE ENGINEERING")
print("="*70)

features_df = features_df \
    .withColumn(
        "funding_gap_usd",
        col("total_requirements_usd") - col("total_funding_usd")
    ).withColumn(
        "funding_gap_pct",
        when(col("total_requirements_usd") > 0,
             (col("total_requirements_usd") - col("total_funding_usd"))
             / col("total_requirements_usd") * 100
             ).otherwise(lit(None))
    ).withColumn(
        "funding_coverage_rate",
        col("total_funding_usd") / (col("total_requirements_usd") + lit(1e-9))
    ).withColumn(
        "beneficiary_to_funding_ratio",
        col("total_targeted") / (col("total_funding_usd") + lit(1e-9))
    ).withColumn(
        "need_to_requirements_ratio",
        col("total_inneed") / (col("total_requirements_usd") + lit(1e-9))
    ).withColumn(
        "targeting_coverage_rate",
        when(col("total_inneed") > 0,
             col("total_targeted") / col("total_inneed")
             ).otherwise(lit(None))
    ).withColumn(
        "cost_per_beneficiary",
        when(col("total_targeted") > 0,
             col("total_funding_usd") / col("total_targeted")
             ).otherwise(lit(None))
    ).withColumn(
        "funding_per_capita",
        when(col("total_population") > 0,
             col("total_funding_usd") / col("total_population")
             ).otherwise(lit(None))
    ).withColumn(
        "requirement_per_capita",
        when(col("total_population") > 0,
             col("total_requirements_usd") / col("total_population")
             ).otherwise(lit(None))
    ).withColumn(
        "disaster_severity_score",
        (coalesce(col("num_disasters_5yr"), lit(0)) * 0.4) +
        (coalesce(col("total_disaster_deaths_5yr"), lit(0)) / lit(10000) * 0.3) +
        (coalesce(col("total_disaster_affected_5yr"), lit(0)) / lit(1000000) * 0.3)
    )

# Sector-level window features
sector_window = Window.partitionBy("country_code", "cluster_name")
features_df = features_df \
    .withColumn("sector_total_funding_usd",
                spark_sum("total_funding_usd").over(sector_window)) \
    .withColumn("sector_total_requirements_usd",
                spark_sum("total_requirements_usd").over(sector_window)) \
    .withColumn("sector_funding_efficiency_pct",
                col("sector_total_funding_usd") /
                (col("sector_total_requirements_usd") + lit(1e-9)) * 100) \
    .withColumn("project_vs_sector_avg",
                col("funding_coverage_rate") -
                (col("sector_total_funding_usd") /
                 (col("sector_total_requirements_usd") + lit(1e-9))))

print("✓ Features engineered:")
feature_cols = [
    "funding_gap_usd", "funding_gap_pct", "funding_coverage_rate",
    "beneficiary_to_funding_ratio", "need_to_requirements_ratio",
    "targeting_coverage_rate", "cost_per_beneficiary",
    "funding_per_capita", "requirement_per_capita",
    "disaster_severity_score", "sector_funding_efficiency_pct",
    "project_vs_sector_avg",
]
for f in feature_cols:
    print(f"  • {f}")

# COMMAND ----------
# ── STEP 9: WRITE FEATURES DELTA TABLE ────────────────────────────────────────
print("\n" + "="*70)
print("STEP 9: WRITING humanitarian.features")
print("="*70)

features_df.cache()
features_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("humanitarian.features")

row_count = features_df.count()
print(f"✓ humanitarian.features: {row_count:,} rows × {len(features_df.columns)} cols")
features_df.unpersist()

# COMMAND ----------
# ── STEP 10: WRITE PROJECTS DELTA TABLE ───────────────────────────────────────
print("\n" + "="*70)
print("STEP 10: WRITING humanitarian.projects")
print("="*70)

proj_prep.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("humanitarian.projects")
print(f"✓ humanitarian.projects: {proj_prep.count():,} rows")

# COMMAND ----------
# ── STEP 11: WRITE CONTRIBUTIONS DELTA TABLE ──────────────────────────────────
contrib_agg.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("humanitarian.contributions")
print(f"✓ humanitarian.contributions: {contrib_agg.count():,} rows")

# COMMAND ----------
# ── VERIFICATION ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

verify = spark.table("humanitarian.features")
verify.select(
    "country_code", "cluster_name", "year",
    "total_requirements_usd", "total_funding_usd",
    "funding_gap_pct", "funding_coverage_rate",
    "beneficiary_to_funding_ratio", "disaster_severity_score"
).show(10, truncate=False)

print("\n✓ Notebook 02 complete")
