import pandas as pd
import os

df_train = pd.read_csv("new_set.csv")

train_set = []
test_set = []

# Loop through breeds
for breed in df_train["breed"].unique():
    breed_df = df_train[df_train["breed"] == breed]
    n = len(breed_df)

    # Index
    split = int(0.8 * n)

    train_set += breed_df[:split].to_dict(orient = "records")
    test_set += breed_df[split:].to_dict(orient = "records")

df_train_set = pd.DataFrame(train_set)
df_test_set = pd.DataFrame(test_set)

df_train_set.to_csv("train_set2.csv", index = False)
df_test_set.to_csv("test_set2.csv", index = False)

