from src.preprocessing import full_preprocessing

X, y, df = full_preprocessing("data/raw/data.csv")

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Preview:\n", df.head())