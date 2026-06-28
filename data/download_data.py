
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()

X = pd.DataFrame(data.data , columns=data.feature_names)
y = pd.Series(data.target, name="price")

df = pd.concat([X, y], axis=1)

df.to_csv("california_housing.csv", index=False)
