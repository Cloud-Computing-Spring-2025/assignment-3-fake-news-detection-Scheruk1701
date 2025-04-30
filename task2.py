from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task2").getOrCreate()

# Step 2: Load CSV again (since it's a separate script)
df = spark.read.option("header", True).option("inferSchema", True).csv("fake_news_sample.csv")

# Step 3: Convert text to lowercase
df_lower = df.withColumn("text_lower", lower(col("text")))

# Step 4: Tokenize text into words
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
df_tokenized = tokenizer.transform(df_lower)

# Step 5: Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Step 6: Select required columns
df_task2 = df_filtered.select("id", "title", "filtered_words", "label")

# Step 7: Save to CSV
df_task2.toPandas().to_csv("task2_output.csv", index=False)

# Optional: Register view
df_task2.createOrReplaceTempView("cleaned_news")
