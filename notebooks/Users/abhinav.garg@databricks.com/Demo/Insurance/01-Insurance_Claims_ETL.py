# Databricks notebook source
# MAGIC %md
# MAGIC ![Insurance Fraud Detection](https://bitrefine.group/images/1920x870/insurance_claim_1920x870.jpg)
# MAGIC 
# MAGIC # Insurance Claims - Fraud Detection
# MAGIC 
# MAGIC In this example, we will be working with some auto insurance data to:
# MAGIC * Do some exploratory analysis and ETL
# MAGIC * Do adhoc Analysis for Reporting
# MAGIC * Demonstrate how to create a predictive model that predicts if an insurance claim is fraudulent or not.

# COMMAND ----------

# MAGIC %md ##1. Data Engineering - Exploratory Analysis & ETL

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/insurance_claims.csv

# COMMAND ----------

# MAGIC %sh head /dbfs/FileStore/tables/insurance_claims.csv

# COMMAND ----------

from pyspark.sql.types import *

schema = (StructType().
          add("months_as_customer", IntegerType()).add("age", IntegerType()).
          add("policy_number", IntegerType()).add("policy_bind_date", StringType()).
          add("policy_state", StringType()).add("policy_csl", StringType()).
          add("policy_deductible", IntegerType()).add("policy_annual_premium", DoubleType()).
          add("umbrella_limit", IntegerType()).add("insured_zip", IntegerType()).
          add("insured_sex", StringType()).add("insured_education_level", StringType()).
          add("insured_occupation", StringType()).add("insured_hobbies", StringType()).
          add("insured_relationship", StringType()).add("capital_gains", IntegerType()).
          add("capital_loss", IntegerType()).add("incident_date", StringType()).
          add("incident_type", StringType()).add("collision_type", StringType()).
          add("incident_severity", StringType()).add("authorities_contacted", StringType()).
          add("incident_state", StringType()).add("incident_city", StringType()).
          add("incident_location", StringType()).add("incident_hour_of_the_day", IntegerType()).
          add("number_of_vehicles_involved", IntegerType()).add("property_damage", StringType()).
          add("bodily_injuries", IntegerType()).add("witnesses", IntegerType()).
          add("police_report_available_", StringType()).add("total_claim_amount", IntegerType()).
          add("injury_claim", IntegerType()).add("property_claim", IntegerType()).
          add("vehicle_claim", IntegerType()).add("auto_make", StringType()).
          add("auto_model", StringType()).add("auto_year", IntegerType()).
          add("class_label", StringType())
         )

# COMMAND ----------

claimsDf = (spark.read.csv("dbfs:/FileStore/tables/insurance_claims.csv", 
                         schema=schema, header=True, 
                         ignoreLeadingWhiteSpace=True, 
                         ignoreTrailingWhiteSpace=True,
                         nullValue='?'))

claimsDf = claimsDf.na.fill({'property_damage': 'NA', 
                             'police_report_available_': 'NA', 
                             "collision_type": 'NA'})

display(claimsDf)

# COMMAND ----------

from pyspark.sql import functions as F

distinctMap = {}
def getDistinctVals(col):
  distinctVals = claimsDf.select(F.collect_set(col).alias('distinct_vals')).first()['distinct_vals']
  distinctMap[col] = distinctVals

# COMMAND ----------

map(getDistinctVals, ['insured_sex','insured_education_level','insured_occupation','insured_relationship','incident_type','collision_type','authorities_contacted'])
print distinctMap

# COMMAND ----------

display(claimsDf.select((claimsDf.months_as_customer/10).cast(IntegerType()).alias("month_bin")).groupBy("month_bin").count().orderBy("month_bin"))

# COMMAND ----------

display(claimsDf.select((claimsDf.total_claim_amount/3000).cast(IntegerType()).alias("amt_bin")).groupBy("amt_bin").count().orderBy("amt_bin"))

# COMMAND ----------

from pyspark.ml.feature import QuantileDiscretizer

claimsDf = (
             claimsDf.
             withColumnRenamed("incident_state", "state_code").
             withColumnRenamed("police_report_available_", "is_police_report_available").
             withColumn("policy_bind_dt", F.to_date(claimsDf.policy_bind_date, 'MM/dd/yy')).
             withColumn("incident_dt", F.to_date(claimsDf.incident_date, 'MM/dd/yy')).
             withColumn("months_as_cust_bin", (claimsDf.months_as_customer/10).cast(IntegerType()))
           )

claimsTransformedDf = claimsDf.drop("policy_bind_date","incident_date")

colNumBucketDict = {"age": 10, "policy_annual_premium": 10, "capital_gains": 30, "capital_loss": 30, "total_claim_amount": 30, "injury_claim": 30, "property_claim": 30, "vehicle_claim": 30}

for col, numBucket in colNumBucketDict.iteritems():
  claimsTransformedDf = QuantileDiscretizer(numBuckets=numBucket, inputCol=col, outputCol=col+"_bin").fit(claimsTransformedDf).transform(claimsTransformedDf)

display(claimsTransformedDf)

# COMMAND ----------

# MAGIC %sql 
# MAGIC   CREATE DATABASE IF NOT EXISTS INSURANCE_DB
# MAGIC   LOCATION "dbfs:/FileStore/databricks-abhinav/insurance"

# COMMAND ----------

# MAGIC %sql USE INSURANCE_DB

# COMMAND ----------

claimsTransformedDf.write.saveAsTable("INSURANCE_CLAIMS_TBL", format = "parquet", mode = "overwrite", partitionBy = "STATE_CODE", path = "dbfs:/FileStore/databricks-abhinav/insurance/claims")

# COMMAND ----------

# MAGIC %sql DESCRIBE INSURANCE_CLAIMS_TBL

# COMMAND ----------

# MAGIC %md ## <div style="float:right"><a href="$./02-Insurance_Claims_AdAnalysis">Ad-hoc Data Analysis</a> <b style="font-size: 160%; color: #1CA0C2;">&#8680;</b></div>