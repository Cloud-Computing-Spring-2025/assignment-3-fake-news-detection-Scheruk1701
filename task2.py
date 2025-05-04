from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FakeNews_Task2_Preprocessing") \
    .getOrCreate()

# Load the dataset from Task 1
df = spark.read.option("header", True).csv("fake_news_sample.csv", inferSchema=True)

# Step 1: Tokenize the text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

# Step 2: Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(tokenized_df)

# Step 3: Convert filtered word arrays to a comma-separated string
def array_to_string(words):
    return ",".join(words)

array_to_string_udf = udf(array_to_string, StringType())
final_df = filtered_df.withColumn("filtered_words_str", array_to_string_udf(col("filtered_words")))

# Step 4: Select and show sample output
final_df.select("id", "title", "filtered_words_str", "label").show(5, truncate=False)

# Step 5: Write result to CSV
# Convert to Pandas and write
final_df.select("id", "title", "filtered_words_str", "label") \
    .toPandas().to_csv("output/task2_output.csv", index=False)

# Stop session
spark.stop()