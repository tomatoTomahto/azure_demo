-- Create a database called Twitter
CREATE DATABASE twitter;
USE twitter;

-- Create a table called tweets, where data resides in ADLS
DROP TABLE IF EXISTS tweets;
CREATE EXTERNAL TABLE IF NOT EXISTS tweets(
    id              STRING,
    tweetDate       TIMESTAMP,
    tweetLanguage   STRING,
    text            STRING,
    userId          STRING,
    keyPhrases      STRING,
    hashtags        ARRAY<STRING>,
    mentions        ARRAY<STRING>,
    cleanText       STRING,
    words           ARRAY<STRING>,
    nonStopWords    ARRAY<STRING>,
    stemmedWords    ARRAY<STRING>
) STORED AS PARQUET LOCATION 'adl://suncordemo.azuredatalakestore.net/Twitter/Processed/tweets';

-- Create a table called users, where data resides in ADLS
DROP TABLE IF EXISTS users;
CREATE EXTERNAL TABLE IF NOT EXISTS users(
    userId          STRING,
    username        STRING,
    locationOfUser  STRING,
    friendsCount    STRING,
    followersCount  STRING
) STORED AS PARQUET LOCATION 'adl://suncordemo.azuredatalakestore.net/Twitter/Processed/users';

-- Whenever new data is loaded into ADLS, run a REFRESH command to refresh Impala's metadata with new file locations
REFRESH tweets;
REFRESH users;

-- Get all users and all tweets
SELECT * FROM tweets;
SELECT * FROM users;

-- Get a count of all tweets and all users
SELECT count(*) as total_tweets FROM tweets;
SELECT count(*) as total_users FROM users;

-- Get a list of all the users and the number of total number of tweets they sent out
SELECT users.username, users.friendsCount, users.followersCount, count(distinct tweets.id) as num_tweets
FROM users, tweets
WHERE users.userId = tweets.userid
GROUP BY users.username, users.friendsCount, users.followersCount
ORDER BY num_tweets DESC;

-- Get the top hashtags used
SELECT hashtags.item AS hashtag, count(*) as numTweets
FROM processed_tweets.hashtags as hashtags
GROUP BY hashtag
ORDER BY numTweets DESC;

-- Get the top user mentions
SELECT mentions.item AS mention, count(*) as numTweets
FROM processed_tweets.mentions as mentions
GROUP BY mention
ORDER BY numTweets DESC;

-- Get the top words (stemmed) used in tweets
SELECT stemmedWords.item AS stemmedWord, count(*) as numTweets
FROM processed_tweets.stemmedWords as stemmedWords
GROUP BY stemmedWord
ORDER BY numTweets DESC;