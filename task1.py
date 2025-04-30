from pyspark.sql import SparkSession

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task1").getOrCreate()

# Step 2: Load the CSV (change path if needed)
df = spark.read.option("header", True).option("inferSchema", True).csv("fake_news_sample.csv")

# Step 3: Create Temp View
df.createOrReplaceTempView("news_data")

# Step 4: Show first 5 rows
df.show(5)

# Step 5: Count total number of articles
total_articles = df.count()
print(f"Total Articles: {total_articles}")

# Step 6: Distinct labels
df.select("label").distinct().show()

# Step 7: Save sample query result to CSV
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)
