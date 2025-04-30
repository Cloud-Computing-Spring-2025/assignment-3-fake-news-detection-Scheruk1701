from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import split, regexp_replace, col, udf
from pyspark.sql.types import BooleanType
from pyspark.ml.linalg import VectorUDT

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task3").getOrCreate()

# Step 2: Load tokenized output from Task 2
df = spark.read.option("header", True).option("inferSchema", True).csv("task2_output.csv")

# Step 3: Convert filtered_words to array
df = df.withColumn("filtered_words", split(regexp_replace("filtered_words", r"[\[\]']", ""), ",\\s*"))

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

# Optional Step 7: Remove empty vectors
def is_non_empty(v):
    return v is not None and v.numNonzeros() > 0

non_empty_udf = udf(is_non_empty, BooleanType())
df_final = df_final.filter(non_empty_udf("features"))

# Step 8: Save in Parquet format
df_final.select("id", "filtered_words", "features", "label_index") \
        .write.mode("overwrite").parquet("task3_output.parquet")
