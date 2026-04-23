import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("cropdata_updated.csv")

# Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_stage = LabelEncoder()

df["crop ID"] = le_crop.fit_transform(df["crop ID"])
df["soil_type"] = le_soil.fit_transform(df["soil_type"])
df["Seedling Stage"] = le_stage.fit_transform(df["Seedling Stage"])

# Features & target
X = df.drop("result", axis=1)
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save everything (model + encoders)
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "le_crop": le_crop,
        "le_soil": le_soil,
        "le_stage": le_stage
    }, f)

print("Model saved as model.pkl")