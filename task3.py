from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import split, col, udf
from pyspark.sql.types import BooleanType, StringType

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task3").getOrCreate()

# Step 2: Load tokenized output from Task 2
df = spark.read.option("header", True).option("inferSchema", True).csv("output/task2_output.csv")

# Step 3: Convert filtered_words_str (comma-separated) to array
df = df.withColumn("filtered_words", split(col("filtered_words_str"), ","))

# Step 4: Term Frequency
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_tf = hashingTF.transform(df)

# Step 5: Inverse Document Frequency
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)

# Step 6: Label Indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)

# Step 7: Filter out empty vectors
def is_non_empty(v):
    return v is not None and v.numNonzeros() > 0

non_empty_udf = udf(is_non_empty, BooleanType())
df_final = df_final.filter(non_empty_udf("features"))

# Step 8: Convert vector to string for saving
vector_to_string_udf = udf(lambda v: str(v), StringType())
df_final = df_final.withColumn("features_str", vector_to_string_udf("features"))

# Step 9: Save output
df_final.select("id", "filtered_words_str", "features_str", "label_index") \
        .write.option("header", True) \
        .mode("overwrite") \
        .csv("output/task3_output.csv")

# Stop Spark session
spark.stop()