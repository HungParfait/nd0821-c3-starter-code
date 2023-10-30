from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import hydra

from ml.data import process_data
from ml.model import inference


app = FastAPI()
MODEL_PATH = "model/lr_model.pkl"
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

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 32,
                    "workclass": "Private",
                    "fnlgt": 205019,
                    "education": "Assoc-acdm",
                    "education-num": 12,
                    "marital-status": "Never-married",
                    "occupation": "Sales",
                    "relationship": "Not-in-family",
                    "race": "Black",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 50,
                    "native-country": "United-States"
                }
            ]
        }
    }


@app.get(path="/")
def welcome():
    return {"message": "Welcome!"}


@app.post(path="/inference")
# @hydra.main(config_path=".", config_name="config", version_base="1.2")
async def prediction(input_data: InputData) -> Dict[str, str]:
    """
    Returning model output from POST request.
    Inputs:
        input_data: InputData
            Instance of a InputData object.
    Returns:
        dict: Dictionary containing the model output.
    """
    [encoder, lb, model] = pickle.load(
        open(MODEL_PATH, "rb"))
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict(by_alias=True).items()}, index=[0]
    )

    processed_input_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, processed_input_data)
    return {"Output": ">50K" if int(prediction[0]) == 1 else "<=50K"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=5000)
