# # Overview
# This script cleanses raw historian/sensor data stored in Azure Data Lake Store
# This script requires the following environment variables to be set:
# * ADLS_PATH - name of the adls store (ie. adl://myadls.azuredatalakestore.net)
# * RAW_HIST_DATA_PATH - directory containing the raw historian data (appended to ADLS_PATH)
# * PROC_HIST_DATA_PATH - directory to store the processed historian data (appended to ADLS_PATH)

# The script uses Spark (via PySpark libraries) to do read the data and process it across
# multiple nodes in a cluster. Once the data is cleansed/processed, it is writte back out
# to ADLS. The data stored there can then be further analyzed/processed by data scientists,
# engineers, or analysts. The figure below illustrates a typical workflow from data ingest
# to data analysis

# # Initialization
# Import relevent Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

# Import relevant python libraries
from datetime import datetime as dt
import os

# Current timestamp for error handling
runDay = dt.now().strftime('%Y-%m-%d')

# Create a Spark Session
spark = SparkSession\
  .builder\
  .appName("BatchProcessHistorianData")\
  .getOrCreate()
  
# Location of raw historian data in ADLS 
adls = os.environ["ADLS_PATH"]
rawDataDir = os.path.join(adls,os.environ['RAW_HIST_DATA_PATH'])
processedDataDir = os.path.join(adls,os.environ['PROC_HIST_DATA_PATH'],runDay)
cleanDataDir = os.path.join(processedDataDir,'clean')
errorDir = os.path.join(processedDataDir,'errors')

# # Load Data into Spark
# Read in raw data from ADLS and enrich the data with the filename were each record came from
rawData = spark.read.load(rawDataDir, format="csv")\
  .withColumn('filename', F.input_file_name()).coalesce(20)

# Persist this data in memory for faster processing
rawData.persist(StorageLevel.MEMORY_ONLY)
  
# # Data Cleansing
# Here we'll demonstrate data cleansing capabilities in Spark. 
# Let's start with simply showing the first 20 records
rawData.show(20, truncate=False)

# It seems that we have 2 columns, one with timestamps and the other with readings
# Let's name those columns and add in a column for the filename (device ID) the record comes from
cleansedData = rawData.withColumn('recordTime',F.to_timestamp('_c0','MM/dd/yyyy HH:mm:ss'))\
  .withColumn('value',rawData._c1.cast('float'))\
  .withColumn('deviceID',F.regexp_extract('filename',rawDataDir+'/PI_[A-Z]+_(.*).csv',1))
  
# Let's filter out the errors
# * non-float or null values
# * non-standard or empty timestamps
# * (optional) values <= 0
valueErrors = cleansedData.filter(cleansedData.value.isNull())\
  .withColumn('Error',F.lit('Unexpected Value'))

timestampErrors = cleansedData.filter(cleansedData.recordTime.isNull())\
  .withColumn('Error',F.lit('Unexpected Timestamp'))

deviceErrors = cleansedData.filter(cleansedData.deviceID=='')\
  .withColumn('Error',F.lit('Non-Standard Device Name'))
  
#readingErrors = cleansedData.filter(cleansedData.value<=0)\
#  .withColumn('Error',F.lit('Non-Positive Reading Value'))

errors = valueErrors.union(timestampErrors).union(deviceErrors)#.union(readingErrors)

# ## Total Errors Found
errors.persist(StorageLevel.MEMORY_ONLY)
errors.groupBy('Error').count().show()

cleansedData = cleansedData.subtract(errors.drop('Error'))

# # Data Extract to ADLS
# Write errors to ADLS so they can be processed later
errors.select('_c0','_c1','Error').coalesce(20)\
  .write.csv(errorDir, mode='overwrite')

# Split the cleansed data into 2 data sets - one for training models and one for testing them
(train, test) = cleansedData.randomSplit([.7,.3])

train.select('recordTime','value','deviceID').dropDuplicates().coalesce(20)\
  .write.parquet(os.path.join(cleanDataDir,'train'), mode='overwrite', compression='snappy')
  
test.select('recordTime','value','deviceID').dropDuplicates().coalesce(20)\
  .write.parquet(os.path.join(cleanDataDir,'test'), mode='overwrite', compression='snappy')