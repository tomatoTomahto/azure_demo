CREATE DATABASE historian;
USE historian;

-- TestReadings contains test data for each run (testday)
CREATE EXTERNAL TABLE testreadings (
    recordTime  TIMESTAMP,
    value       FLOAT,
    deviceID    STRING
) PARTITIONED BY (testday STRING) STORED AS PARQUET;

-- TrainingReadings contains training data for each run (trainingday)
CREATE EXTERNAL TABLE trainingreadings (
    recordTime  TIMESTAMP,
    value       FLOAT,
    deviceID    STRING
) PARTITIONED BY (trainingday STRING) STORED AS PARQUET;

-- ScoredReadings contains scored data for each run (predictionday) and model
CREATE EXTERNAL TABLE scoredreadings (
    recordInterval  TIMESTAMP,
    deviceID        STRING,
    reading         DOUBLE,
    prediction      DOUBLE,
    model           STRING
) PARTITIONED BY (predictionday STRING) STORED AS PARQUET;

-- Compute the average residual for each model run
SELECT * FROM scoredreadings;
SELECT model, predictionday, AVG(reading-prediction) AS residual
FROM scoredreadings
GROUP BY model, predictionday
ORDER BY predictionday ASC;