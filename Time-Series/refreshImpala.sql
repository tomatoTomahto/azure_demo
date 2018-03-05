USE historian;

ALTER TABLE trainingReadings ADD IF NOT EXISTS PARTITION (trainingDay='${var:day}') LOCATION 'adl://suncordemo.azuredatalakestore.net/CleanHistorical/${var:day}/clean/train';
REFRESH trainingReadings;

ALTER TABLE testReadings ADD IF NOT EXISTS PARTITION (testDay='${var:day}') LOCATION 'adl://suncordemo.azuredatalakestore.net/CleanHistorical/${var:day}/clean/test';
REFRESH testReadings;

ALTER TABLE scoredReadings ADD IF NOT EXISTS PARTITION (predictionDay='${var:day}') LOCATION 'adl://suncordemo.azuredatalakestore.net/CleanHistorical/${var:day}/clean/predictions';
REFRESH scoredReadings;