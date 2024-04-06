from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import mean
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField

import pandas as pd


spark = (
    SparkSession
    .builder
    .appName("PySpark Lection 2")
    .master("local")
    .getOrCreate()
)

train_df = pd.read_csv('data/train.csv', usecols=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
                                                  'Street', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'SalePrice'])
test_df = pd.read_csv('data/test.csv', usecols=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
                                                  'Street', '1stFlrSF', '2ndFlrSF', 'OverallQual'])

# Somehow it does not work. Throws `Index out of range` error
train_schema = StructType(fields=[
    StructField("MSSubClass", IntegerType()),
    StructField("MSZoning", StructType()),
    StructField("LotFrontage", FloatType(), nullable=True),
    StructField("LotArea", IntegerType()),
    StructField("Street", StringType()),
    StructField("OverallQual", IntegerType()),
    StructField("1stFlrSF", IntegerType()),
    StructField("2ndFlrSF", IntegerType()),
    StructField("SalePrice", FloatType())
])

test_schema = StructType(fields=[
    StructField("MSSubClass", IntegerType()),
    StructField("MSZoning", StructType()),
    StructField("LotFrontage", FloatType(), nullable=True),
    StructField("LotArea", IntegerType()),
    StructField("Street", StringType()),
    StructField("OverallQual", IntegerType()),
    StructField("1stFlrSF", IntegerType()),
    StructField("2ndFlrSF", IntegerType())
])

train = spark.createDataFrame(train_df)
test = spark.createDataFrame(test_df)

# Показать датафреймы
print(train.show())

# Заполнить в колонке NaN средним значением
# Удалить таргет из трейна (1-е преобразование)

train_without_target = train.drop("SalePrice")

# Объединить датафреймы (2-е преобразование)

combined_df = train_without_target.union(test)

# Получить среднее для колонки (3-е преобразование)

mean_value = combined_df.dropna().select(mean(combined_df['LotFrontage'])).collect()[0][0]

train = train.fillna({'LotFrontage': round(mean_value, 1)})
test = test.fillna({'LotFrontage': round(mean_value, 1)})

# Удалить колонку (4-е преобразование)

train = train.drop("MSZoning")
test = test.drop('MSZoning')

# Переименуем колонку (5-е преобразование)

train = train.withColumnRenamed("MSSubClass", "DwellingClass")
test = test.withColumnRenamed("MSSubClass", "DwellingClass")

# Покажем в train только те дома, у которых `Overall Quality` = 10 (6-е преобразование)

print(train.filter(train.OverallQual == 10).show())

# One-Hot Encoding (7-е и тд преобразование)

categorical_col = "Street"
string_indexer = StringIndexer(inputCol=categorical_col, outputCol="indexed_" + categorical_col)
encoder = OneHotEncoder(inputCol="indexed_" + categorical_col, outputCol="onehot_" + categorical_col)

pipeline = Pipeline(stages=[string_indexer, encoder])
pipeline_model = pipeline.fit(train)
train = pipeline_model.transform(train)

test = pipeline_model.transform(test)
train = train.drop("Street", "indexed_Street")
test = test.drop("Street", "indexed_Street")

# Линейная регрессия

assembler = VectorAssembler(
    inputCols=["DwellingClass", "LotFrontage", "LotArea", "OverallQual", "1stFlrSF", "2ndFlrSF", "onehot_Street"],
    outputCol="features")

train_features = assembler.transform(train)
test_features = assembler.transform(test)

lr = LinearRegression(featuresCol="features", labelCol="SalePrice")
lr_model = lr.fit(train_features)

predictions = lr_model.transform(test_features)
train_preds = lr_model.transform(train_features)

# Evaluate predictions using regression evaluator
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="mse")
mse = evaluator.evaluate(train_preds)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="rmse")
rmse = evaluator.evaluate(train_preds)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="mae")
mae = evaluator.evaluate(train_preds)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SalePrice", metricName="r2")
r2 = evaluator.evaluate(train_preds)

print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("Mean Absolute Error (MAE):", round(mae, 3))
print("R-squared (R²):", round(r2, 3))
