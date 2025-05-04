from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import split

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task4") \
    .getOrCreate()

# Step 2: Load the CSV generated from Task 2
df = spark.read.option("header", "true").csv("output/task2_output.csv", inferSchema=True)

# Step 3: Preprocess text - split filtered string into array
df = df.withColumn("filtered_words_array", split(df["filtered_words_str"], " "))

# Step 4: Define Pipeline Stages
hashingTF = HashingTF(inputCol="filtered_words_array", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

pipeline = Pipeline(stages=[hashingTF, idf, label_indexer])

# Step 5: Fit the pipeline and transform the data
pipeline_model = pipeline.fit(df)
transformed_df = pipeline_model.transform(df)

# Step 6: Split into train and test sets
train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

# Step 7: Train Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_df)

# Step 8: Make predictions
predictions = lr_model.transform(test_df)

# Step 9: Save predictions to CSV
predictions.select("id", "title", "label_index", "prediction") \
    .write.option("header", "true") \
    .mode("overwrite") \
    .csv("output/task4_output.csv")

# Optional: Print some predictions
predictions.select("id", "title", "label_index", "prediction").show(10, truncate=False)

# Step 10: Stop Spark session
spark.stop()
