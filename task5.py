from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import pandas as pd
import os

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task5").getOrCreate()

# Step 2: Load predictions from Task 4
df = spark.read.option("header", True).option("inferSchema", True).csv("output/task4_output.csv")

# Step 3: Cast prediction and label columns to double if needed
df = df.withColumn("label_index", col("label_index").cast("double"))
df = df.withColumn("prediction", col("prediction").cast("double"))

# Step 4: Define evaluators
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)

# Step 5: Compute metrics
accuracy = evaluator_acc.evaluate(df)
f1_score = evaluator_f1.evaluate(df)

# Step 6: Save metrics to CSV (output folder)
os.makedirs("output", exist_ok=True)
output_pd = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [round(accuracy, 4), round(f1_score, 4)]
})
output_pd.to_csv("output/task5_output.csv", index=False)

# Step 7: Print as markdown table
print("\nModel Evaluation Metrics\n")
print(output_pd.to_markdown(index=False))

# Stop Spark session
spark.stop()
