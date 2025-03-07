import sys
import pytest
from fastapi.testclient import TestClient
sys.path.append('./')
from main import app


@pytest.fixture(scope="session")
def client():
    client = TestClient(app)
    return client


def test_get(client):
    """Test standard get"""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Welcome!"}


def test_post_above_50K(client):
    """Test for salary above 50K"""

    res = client.post("/inference", json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 100,
        "hours-per-week": 50,
        "native-country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'Output': '>50K'}


def test_post_below_50K(client):
    """Test for salary below 50K"""
    res = client.post("/inference", json={
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    })

    assert res.status_code == 200
    assert res.json() == {'Output': '<=50K'}


def test_get_invalid_url(client):
    """Test invalid url"""
    res = client.get("/invalid_url")
    assert res.status_code == 404
    assert res.json() == {'detail': 'Not Found'}
