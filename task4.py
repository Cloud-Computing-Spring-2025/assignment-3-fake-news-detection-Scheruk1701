from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task4").getOrCreate()

# Step 2: Load output from Task 3 (Parquet preserves schema)
df = spark.read.parquet("task3_output.parquet")

# Step 3: Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Step 4: Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_df)

# Step 5: Make predictions
predictions = model.transform(test_df)

# Step 6: Load titles for joining
titles_df = spark.read.option("header", True).csv("fake_news_sample.csv")

# Step 7: Join to get readable titles
result_df = predictions.join(titles_df.select("id", "title"), on="id", how="left")

# Step 8: Save results
result_df.select("id", "title", "label_index", "prediction") \
         .toPandas().to_csv("task4_output.csv", index=False)
