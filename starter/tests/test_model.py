import pickle
import pandas as pd
import pandas.api.types as pdtypes
import pytest
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="module")
def data():
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)


def test_type_and_presence(data):
    """Tests that cleaned csv file has expected columns and types.

    Inputs:
        data (pd.DataFrame): Dataset for testing
    """

    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def workclass_values(data):
    """Tests that the workclass column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    }

    assert set(data["workclass"].unique()) == expected_values


def education_values(data):
    """Tests that the education column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    }

    assert set(data["education"].unique()) == expected_values


def marital_status_values(data):
    """Tests that the marital-status column has the expected values.

    Args:
        data (pd.DataFrame): Dataset for testing
    """
    expected_values = {
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }

    assert set(data["marital-status"].unique()) == expected_values
