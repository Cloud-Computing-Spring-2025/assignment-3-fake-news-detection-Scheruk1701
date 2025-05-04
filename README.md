# 📘 Assignment-5-FakeNews-Detection

## 📁 Dataset
The dataset used in this project is generated using the `Dataset_Generator.py` script. It utilizes the `Faker` library to simulate fake and real news articles, outputting a labeled dataset in CSV format:  
📄 `fake_news_sample.csv`

---

## 🧠 Overview

This assignment walks through the end-to-end process of detecting fake news using natural language processing and machine learning with **Apache Spark**.

We implement the following tasks in sequential steps:

### 🧼 Task 1: Data Cleaning
- Reads the generated CSV file.
- Cleans the text by removing punctuation, digits, stopwords, and lowercasing.
- Saves the cleaned result into `output/task1_output.csv`.

### 🪄 Task 2: Tokenization and Lemmatization
- Tokenizes the cleaned text.
- Applies lemmatization using NLTK’s WordNetLemmatizer.
- Stores filtered tokens back as a string column.
- Output: `output/task2_output.csv`.

### 🧮 Task 3: Feature Extraction (TF-IDF)
- Splits the filtered string into a word array.
- Applies `HashingTF` followed by `IDF` to get TF-IDF features.
- Uses `StringIndexer` to encode labels ("FAKE", "REAL") into numerical indices.
- Saves the feature matrix and labels to `output/task3_output.csv`.

### 🤖 Task 4: Model Training & Prediction
- Uses `LogisticRegression` on the TF-IDF features.
- Trains the model on 80% of the dataset and evaluates on 20%.
- Outputs predictions to `output/task4_output.csv`.

### 📊 Task 5: Evaluation
- Computes performance metrics using predictions from Task 4.
- Stores the metrics in `output/task5_output.csv`.

📈 **Sample Evaluation Output**:
```

Metric,Value
Accuracy,0.4459
F1 Score,0.3204

````

---

## 🚀 How to Run

### 1️⃣ Generate Dataset
```bash
python Dataset_Generator.py
````

### 2️⃣ Run Each Task

```bash
spark-submit task1.py
spark-submit task2.py
spark-submit task3.py
spark-submit task4.py
spark-submit task5.py
```

---

## 📂 Folder Structure

```
.
├── Dataset_Generator.py
├── fake_news_sample.csv
├── output/
│   ├── task1_output.csv
│   ├── task2_output.csv
│   ├── task3_output.csv
│   ├── task4_output.csv
│   └── task5_output.csv
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── task5.py
└── README.md
```

---

## ✅ Prerequisites

Make sure to install the following Python packages:

```bash
pip install pyspark faker
```
---
