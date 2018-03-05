# # Overview
# This script tests out a given linear regression model stored in Azure Data Lake Store 
# against testing data also stored in Azure Data Lake
# This script requires the following environment variables to be set:
# * ADLS_PATH - name of the adls store (ie. adl://myadls.azuredatalakestore.net)
# * PROC_HIST_DATA_PATH - directory to store the processed historian data (appended to ADLS_PATH)
# * MODEL_DEV_PATH - directory to store ML models to be tested

# # Initialization
# Import relevant Spark and Python Libraries
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import PipelineModel
import os, sys
from datetime import datetime as dt
sys.path.append('Time-Series')
import UtilFunctions as util

runDay = dt.now().strftime('%Y-%m-%d')

# Connect to Spark
spark = SparkSession\
  .builder\
  .appName("TestModel")\
  .getOrCreate()
  
# # Test Data Import
# Location of raw historian data in ADLS 
adls = os.environ["ADLS_PATH"]
testDataDir = os.path.join(adls,os.environ['PROC_HIST_DATA_PATH'],runDay,'clean/test')
predDataDir = os.path.join(adls,os.environ['PROC_HIST_DATA_PATH'],runDay,'clean/predictions')
modelDevDir = os.path.join(adls,os.environ["PROC_HIST_DATA_PATH"],runDay,'models')
modelName = 'spark-lrPipelineModel-v3'

testData = spark.read.load(testDataDir, format="parquet")

# # Data Feature Preparation
# This model requires data to be stored 'hourly' and padded for missing hours with previous values
hourlyData = testData.withColumn('recordInterval',(F.round(F.unix_timestamp('recordTime')/3600)*3600).cast('timestamp'))\
  .groupBy('recordInterval','deviceID').agg(F.avg('value').alias('reading'))

# 'Pad' the data by filling in any missing hours using the last good (ie. non-null) value from the device
paddedHourlyData = util.padData(hourlyData, spark)  

# We need the following features:
# * Previous hour's reading
# * Average of last 5 hours reading
window = Window.partitionBy("deviceID").orderBy("recordInterval")
modelData = paddedHourlyData\
  .withColumn('lastReading',F.avg('reading').over(window.rowsBetween(-1,-1)))\
  .withColumn('avgReadings5hr',F.avg('reading').over(window.rowsBetween(-5,-1)))\
  .filter('deviceID=="85WIC2703_COR"')\
  .na.drop()  

# # Model Testing
# Load the entire pipeline (including feature transformations and regression model)
pipelineModel = PipelineModel.load(os.path.join(modelDevDir,modelName))

lrModel = pipelineModel.stages[1]

util.cdswPrint(modelName + ' Model Attributes','h1')
util.cdswPrint("Coefficients: %s" % str(lrModel.coefficients))
util.cdswPrint("Intercept: %s" % str(lrModel.intercept))

# Use the model to score the test data
predictions = pipelineModel.transform(modelData)

# # Model Visualization
# Plot the residual distribution and actual vs predicted readings
predictions = predictions.withColumn('residual', predictions.reading-predictions.prediction)\
  .withColumn('model',F.lit(modelName))
util.plotDist(predictions.select('residual').toPandas(),'Residuals')
util.plotTS(predictions.select('recordInterval','reading','prediction'),'Actual vs. Predicted Readings')

# # Data Export
# Output the predictions to ADLS to be analyzed by other tools
predictions.select('recordInterval','deviceID','reading','prediction','model').coalesce(10)\
  .write.parquet(predDataDir, mode='append', compression='snappy')