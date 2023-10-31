import requests

# Request to the Render server
url = "https://hungnd.onrender.com/inference"

response = requests.post(
    url=url,
    json={
        "age": 59,
        "workclass": "Private",
        "fnlgt": 109015,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
)

print("Status code: " + str(response.status_code))
print("Data: ")
print(response.json())
