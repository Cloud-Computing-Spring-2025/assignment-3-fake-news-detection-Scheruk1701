from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Step 1: Start Spark Session
spark = SparkSession.builder.appName("FakeNews_Task5").getOrCreate()

# Step 2: Load predictions from Task 4
df = spark.read.option("header", True).option("inferSchema", True).csv("task4_output.csv")

# Step 3: Define evaluators
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

# Step 4: Compute metrics
accuracy = evaluator_acc.evaluate(df)
f1_score = evaluator_f1.evaluate(df)

# Step 5: Save to CSV
output_pd = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [round(accuracy, 4), round(f1_score, 4)]
})
output_pd.to_csv("task5_output.csv", index=False)

# Step 6: Print as markdown table
print("\nModel Evaluation Metrics\n")
print(output_pd.to_markdown(index=False))
