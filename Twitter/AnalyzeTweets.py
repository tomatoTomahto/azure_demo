# # Overview
# This script reads processed Twitter data from Azure Data Lake Store and analyzes it
# using Spark and Python APIs. The following environment variables should be set before running
# the script:
# * ADLS_PATH - adls store (ie. adl://myadlsstore.azuredatalakestore.net)
# * TWITTER_PROCESSED_PATH - directory in ADLS to write processed Twitter data

# # Initialization
# ## Import relevant Spark and Python libraries
from pyspark.sql import SparkSession, functions as F, Window
from pyspark import StorageLevel
from pyspark.sql.types import *
from pyspark.ml.feature import NGram

from datetime import datetime as dt
import os
import seaborn as sb
import matplotlib.pyplot as plt

# ## Connect to the Spark cluster
spark = SparkSession\
  .builder\
  .appName("AnalyzeTweets")\
  .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
  
# # Data Import
# Read in raw data from ADLS
adls = os.environ['ADLS_PATH']
dataDir = os.path.join(adls,os.environ['TWITTER_PROCESSED_PATH'])

tweets = spark.read.load(os.path.join(dataDir,'tweets'), format="parquet")
tweets.printSchema()

# # Data Analysis
# ## Top 20 Hashtags, User Mentions, Words, and NGrams
def plotTopXEntity(tweets, entity, x):
  data = tweets.select(F.explode(entity).alias(entity),'id')\
    .groupBy(entity).agg(F.countDistinct('id').alias('count')).orderBy('count',ascending=False)\
    .limit(x).toPandas()
    
  sb.set(style="whitegrid")
  f, ax = plt.subplots(figsize=(6, 6))
  sb.set_color_codes("pastel")
  sb.barplot(x="count", y=entity, data=data, color="b")
  ax.set(ylabel="",xlabel="Tweet Count by %s" % entity)
  sb.despine(left=True, bottom=True)

plotTopXEntity(tweets, 'hashtags', 20)
plotTopXEntity(tweets, 'mentions', 20)

# Use Spark's nGram transformer to quickly compute ngrams
ngram = NGram(n=3, inputCol="stemmedWords", outputCol="ngrams")
ngramDataFrame = ngram.transform(tweets)

plotTopXEntity(tweets, 'stemmedWords', 20)
plotTopXEntity(ngramDataFrame, 'ngrams', 20)