# # Overview
# This script reads raw Twitter JSON documents from Azure Data Lake Store and processes them
# using Spark and Python APIs. The following environment variables should be set before running
# the script:
# * ADLS_PATH - adls store (ie. adl://myadlsstore.azuredatalakestore.net)
# * TWITTER_RAW_PATH - directory in ADLS containing raw Twitter JSON documents
# * TWITTER_PROCESSED_PATH - directory in ADLS to write processed Twitter data

# The script will read in data from ADLS, perform some basic filtering and projections to
# extract the fields we are specifically interested in (ie. user info and tweet info), then
# perform natural language processing on the actual tweet text. It will write out the resulting
# data to ADLS. 

# # Initialization
# ## Import relevant Spark and Python libraries
from pyspark.sql import SparkSession, functions as F, Window
from pyspark import StorageLevel
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover

from nltk.stem.porter import *
from datetime import datetime as dt
import os

# ## Connect to the Spark cluster
spark = SparkSession\
  .builder\
  .appName("ProcessTweets")\
  .getOrCreate()
  
# # Data Import
# Read in raw data from ADLS
adls = os.environ['ADLS_PATH']
rawDataDir = os.path.join(adls,os.environ['TWITTER_RAW_PATH'])
cleanDataDir = os.path.join(adls,os.environ['TWITTER_PROCESSED_PATH'])

rawTweets = spark.read.load(rawDataDir, format="json")
rawTweets.printSchema()

# Persist as much data as possible to available memory in the cluster
rawTweets.persist(StorageLevel.MEMORY_ONLY)
rawTweets.show()

# # Transform Data
# We will perform the following transformations on the raw data:
# * Extract relevant user information
# * Extract relevant tweet information
# * Filter out any errors
# * Natural language processing of tweets

# ## Step 1. Convert the timestamp string to a timestamp type so we can perform windowing aggregations
processedTweets = rawTweets.withColumn('tweetDate',F.to_timestamp('tweetDate', "yyyy-MM-dd'T'HH:mm:ss'.000Z'"))
processedTweets.select('tweetDate').show(truncate=False)

# ## Step 2. Extract the User Information from each tweet
# Each tweet has the following user information attached to it as shown in the schema (at the time of the tweet)
# * userId
# * username
# * nameOfUser
# * locationOfUser 
# * friendsCount
# * followersCount
userAttributes = ['tweetDate','userId','username','locationOfUser','friendsCount','followersCount','tweetDate']
users = processedTweets.select(userAttributes)

# Twitter updates the number of friends and followers each time the user makes a tweet, 
# so let's get the latest information for each user using Spark windowing
window = Window.partitionBy("userId").orderBy(F.desc("tweetDate"))
users = users.select(userAttributes + [F.rank().over(window).alias('rank')])\
  .filter('rank==1')\
  .drop('rank','tweetDate')

users.persist(StorageLevel.MEMORY_ONLY)
users.show()  

# ## Step 3. Extract relevant tweet attributes
# Let's grab attributes that we are interested in:
# * id
# * tweetDate
# * tweetLanguage
# * text
# * userId
# * keyPhrases
tweetAttributes = ['id','tweetDate','tweetLanguage','text','userId','keyPhrases']
tweets = processedTweets.select(tweetAttributes)

tweets.persist(StorageLevel.MEMORY_ONLY)
tweets.printSchema()
tweets.show()

# ## Step 4. Quality Check
# We need to ensure that each user has:
# * A unique, non-null, numeric ID

# And that each tweet has:
# * A unique, non-null numeric ID
# * A non-null tweetDate
# * a non-null text
users.describe(['userId','friendsCount','followersCount']).show()
tweets.describe(['id']).show()

users.filter(users.userId.isNull()).show()
tweets.filter(tweets.id.isNull() | tweets.tweetDate.isNull() | tweets.text.isNull()).show()

# ## Step 5. Clean tweet text
# We are interested in performing the following transformations on the raw text of each tweet:
# * Remove punctuations
# * Extract hashtags (#clouderarocks) and user mentions (ie. @cloudera)
tweets.select('text').show()

# UDF to remove all the punctuation and emoticons from a tweet, and convert to lower case
def cleanText(text):
  cleanText = text.lower()

  # Remove emoji's
  emojiPattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
  cleanText = emojiPattern.sub(r'', cleanText)
  
  for remove in map(lambda r: re.compile(re.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                         "%", "^", "*", "(", ")", "{", "}",
                                                         "[", "]", "|", "/", "\\", ">", "<", "-",
                                                         "!", "?", ".", "'","`","’",
                                                         "--", "---"]):
      cleanText = remove.sub(r'', cleanText)
  return cleanText

cleanTextUDF = F.udf(cleanText, StringType())
  
# UDF to extract entities given a regex (ie. used for hashtags (#) and mentions (@))
def extractEntities(text, regex):
  hashtags = re.findall(regex, text.lower())
  return hashtags

extractEntitiesUDF = F.udf(extractEntities, ArrayType(StringType()))
hashtagRegex = "(#[a-zA-Z0-9]{3,15})"
mentionRegex = "(@[a-zA-Z0-9]{3,15})"

cleanTweets = tweets.withColumn('hashtags',extractEntitiesUDF('text',F.lit(hashtagRegex)))\
  .withColumn('mentions',extractEntitiesUDF('text',F.lit(mentionRegex)))\
  .withColumn('cleanText',cleanTextUDF('text'))

cleanTweets.select('text','cleanText','hashtags','mentions').show(5)

# ## Step 6. Tokenize tweet text
# We now want to take the cleansed tweet text and transform it into an array of tokens. To 
# do this, we need to:
# * Tokenize each tweet text
# * Remove any stop words in the text
# * Stem any remaining words in the text

# We will use Spark NLP functions to tokenize and remove stopwords, and NLTK to stem the words
tokenizer = Tokenizer(inputCol="cleanText", outputCol="words")
tokenizedTweets = tokenizer.transform(cleanTweets)

swRemover = StopWordsRemover(inputCol="words", outputCol="nonStopWords")
tokenizedTweetsNoSW = swRemover.transform(tokenizedTweets)

def stem(words):
  stemmer = PorterStemmer()
  stemmed = [stemmer.stem(word) for word in words]
  return stemmed

stemUDF = F.udf(stem,ArrayType(StringType()))
stemmedTweets = tokenizedTweetsNoSW.withColumn('stemmedWords',stemUDF('nonStopWords'))

stemmedTweets.select('text','stemmedWords').show(5)

# # Data Export
# ## Write Cleansed/Processed Data to ADLS
users.coalesce(10)\
  .write.parquet(os.path.join(cleanDataDir,'users'),mode='overwrite', compression='snappy')
stemmedTweets.coalesce(10)\
  .write.parquet(os.path.join(cleanDataDir,'tweets'),mode='overwrite', compression='snappy')