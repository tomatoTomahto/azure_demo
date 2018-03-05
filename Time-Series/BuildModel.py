# # Overview
# This script transforms clean historian data stored in Azure Data Lake Store and builds
# a simple linear regression model to predict the value from a given tag (device ID)
# This script requires the following environment variables to be set:
# * ADLS_PATH - name of the adls store (ie. adl://myadls.azuredatalakestore.net)
# * PROC_HIST_DATA_PATH - directory to store the processed historian data (appended to ADLS_PATH)
# * MODEL_DEV_PATH - directory to store ML models to be tested

# The script uses Spark (via PySpark libraries) to do read the data and process it across
# multiple nodes in a cluster. Various visualizations are done using standard, open Python
# plotting libraries to explore the data. Spark MLlib is used to build a linear regression
# model, which is saved to ADLS for testing and tuning. 

# # Initialization
# Import relevant Spark Libraries
from pyspark.sql import SparkSession, functions as F, Window, Row
from pyspark import StorageLevel
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Import relevant Python libraries
import datetime as dt, os, sys
sys.path.append('Time-Series')
import UtilFunctions as util

runDay = dt.datetime.now().strftime('%Y-%m-%d')

# Connect to the Spark cluster
spark = SparkSession\
  .builder\
  .appName("AnalyzeHistorianData")\
  .getOrCreate()
  
# Location of raw historian data in ADLS 
adls = os.environ["ADLS_PATH"]
trainDataDir = os.path.join(adls,os.environ['PROC_HIST_DATA_PATH'],runDay,'clean/train')
modelDir = os.path.join(adls,os.environ["PROC_HIST_DATA_PATH"],runDay,'models')
modelName = 'spark-lrPipelineModel-v3'

# # Data Import from ADLS
# Read in raw data from ADLS
trainData = spark.read.load(trainDataDir, format="parquet")

# Aggregate the data to the hourly level
hourlyData = trainData.withColumn('recordInterval',(F.round(F.unix_timestamp('recordTime')/3600)*3600).cast('timestamp'))\
  .groupBy('recordInterval','deviceID').agg(F.avg('value').alias('reading'))

# 'Pad' the data by filling in any missing hours using the last good (ie. non-null) value from the device
paddedHourlyData = util.padData(hourlyData, spark)

# Store the resulting dataset into the available memory - all our analysis will be on this dataset going forward
paddedHourlyData.persist(StorageLevel.MEMORY_ONLY)
paddedHourlyData.show()

# # Data Analysis
# Spark allows users to interact with the data using dataframe operators as shown above
# OR using SQL, as shown below. 
# Here we extract all the measurements from device names beginning with '85WIC'
paddedHourlyData.createTempView('hourlyData')
sensorReadings = spark.sql('SELECT * FROM hourlyData WHERE deviceID = "85WIC2703_COR"')

# We can convert the Spark dataframe to a Pandas dataframe for plotting 
util.plotHist(sensorReadings.withColumn('dayOfWeek',F.date_format('recordInterval', 'E'))\
              .toPandas(), 'reading', 'dayOfWeek')

util.plotDist(sensorReadings.select('reading').toPandas(), 'Average Hourly Readings')

# # Feature Engineering
# Let's see if we can build a model that can predict a reading for a sensor based on past readings
# In order to do this we need to assemble a dataframe that has previous readings next to current readings
# Window functions are a great way to do this. Here we create a window for each deviceID ordered by the timestamp
# We can then use window functions like avg() on that window for each record.
# Here we extract the last hour's reading (previous record) and the average reading over the last 5 hours (previous 5 records)
window = Window.partitionBy("deviceID").orderBy("recordInterval")
modelData = paddedHourlyData\
  .withColumn('lastReading',F.avg('reading').over(window.rowsBetween(-1,-1)))\
  .withColumn('avgReadings5hr',F.avg('reading').over(window.rowsBetween(-5,-1)))\
  .filter('deviceID=="85WIC2703_COR"')\
  .na.drop()
modelData.show()

util.plotTS(modelData, 'Readings vs. 1hr Rolling Avg Readings')

# Spark let's us easily calculate the correlation between different columns in the dataframe
corr1 = modelData.corr('reading','lastReading')
corr2 = modelData.corr('reading','avgReadings5hr')
util.cdswPrint('Correlations: [ %2.3f , %2.3f ]' % (corr1, corr2))

# Model Training
# We want to build a simple linear regression model that predicts a sensors current reading
# based on it's previous hourly reading AND it's average hourly reading for the last 5 hours
# We first assemble a vector with the predictive columns, and call that vector 'features'
# Then we can fit a linear regression model on the test data that we imported

va = VectorAssembler(inputCols=["lastReading","avgReadings5hr"], outputCol="features")
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, 
                      labelCol='reading', featuresCol='features')

pipeline = Pipeline(stages=[va,lr])
pipelineModel = pipeline.fit(modelData)

util.cdswPrint('Model Statistics','h1')
util.lrModelStats(pipelineModel.stages[1])

# # Model Export
# We are happy with our model, let's save it to development for further iterations
# We can either export this model in a variety of ways, for example in PMML format
from jpmml_sparkml import toPMMLBytes
pmml = toPMMLBytes(spark.sparkContext, modelData, pipelineModel)
print(str(pmml))

# Or in Spark model format
pipelineModel.save(os.path.join(modelDir,modelName))