# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_on_slices_data
    )
# Add code to load in the data.
DATA_PATH = "../../../data/cencus.csv"
MODEL_PATH = "../../../model"
SLICE_OUTPUT_PATH = "../../../"

data = pd.read_csv(DATA_PATH)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
        X=test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
# Train and save a model.
model = train_model(X_train, y_train)

with open(MODEL_PATH, "wb") as f:
    pickle.dump([encoder, lb, model], f)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

metrics = compute_metrics_on_slices_data(
    df=test,
    cat_columns=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb,
    model=model,
    slice_output_path=SLICE_OUTPUT_PATH
)