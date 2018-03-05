from pyspark.sql import functions as F, Window, Row

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import datetime as dt
import sys
from IPython.core.display import display, HTML

# 'Pad' the data by filling in any missing hours using the last good (ie. non-null) value from the device
def padData(data, spark):
  # To do this, we get a list of all the timestamps between the min and max recordIntervals in the dataset
  min_max = data.agg(F.min('recordInterval'), F.max('recordInterval')).collect()
  minTime = min_max[0][0]
  maxTime = min_max[0][1]
  dateList = [minTime + dt.timedelta(hours=x) for x in range(0, (maxTime-minTime).days*24)]

  # We then cross-join that list with our dataset, so there is a record for every device and timestamp
  dateDF = spark.createDataFrame(spark.sparkContext.parallelize(dateList).map(lambda i: Row(recordInterval=i)))
  padderDF = dateDF.crossJoin(data.select('deviceID').distinct())

  # Lastly, we fill in the missing/null values with the last known good reading from the device using Window functions 
  window = Window.partitionBy('deviceID').orderBy('recordInterval').rowsBetween(-sys.maxsize, 0)
  paddedData = data.join(padderDF, ['recordInterval','deviceID'], 'right') \
    .withColumn("reading", F.last('reading', True).over(window))\
    .na.drop()

  return paddedData

# Plot a histogram from the dataframe provided
def plotHist(data, x, y):
  f, ax = plt.subplots(figsize=(7, 6))
  ax.set_xscale("log")
  sb.boxplot(y=y, x=x, data=data, palette="PRGn", whis=np.inf)
  ax.xaxis.grid(True)
  ax.set(ylabel="", xlabel="Hourly Reading", title='Average Hourly Readings by Day of Week')
  sb.despine(trim=True, left=True)
  plt.show()
  
# Plot a historgram and kernel density estimate
def plotDist(data, title):
  sb.set(style="white", palette="muted", color_codes=True)
  f, ax = plt.subplots(figsize=(7, 7))
  sb.despine(left=True)
  sb.distplot(data, color="m")
  ax.set(title='Distribution of %s' % title)
  plt.setp(ax, yticks=[])
  plt.tight_layout()
  
# Plot a time-series graph for data between a set of days
def plotTS(data, title):
  data.filter((F.to_date('recordInterval').between('2017-08-02 00:00:00','2017-08-03 00:00:00')))\
    .toPandas().plot(x='recordInterval', title=title)
  plt.show()

def cdswPrint(text,t='h2'):
  display(HTML('<%s>%s</%s>' % (t, text, t)))
  
def lrModelStats(lrModel):
  # Spark MLlib provides metrics for all it's models. Here we will output the common ones used for linear regression
  cdswPrint("Coefficients: %s" % str(lrModel.coefficients))
  cdswPrint("Intercept: %s" % str(lrModel.intercept))
  trainingSummary = lrModel.summary
  cdswPrint("numIterations: %d" % trainingSummary.totalIterations)
  print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
  trainingSummary.residuals.describe().show()
  cdswPrint("RMSE: %f" % trainingSummary.rootMeanSquaredError)
  cdswPrint("r2: %f" % trainingSummary.r2)


