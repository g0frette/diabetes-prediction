import requests

# response = requests.get("http://127.0.0.1:8000/health" , timeout = 10)

# print("Status:", response.status_code)

# response.raise_for_status()

# data = response.json()
# print("Server status:", data["status"])


payload = {
    "age": 45,
    "gender": "Male",
    "polyuria": "Yes",
    "polydipsia": "Yes",
    "sudden_weight_loss": "No",
    "weakness": "Yes",
    "polyphagia": "No",
    "genital_thrush": "No",
    "visual_blurring": "No",
    "itching": "No",
    "irritability": "No",
    "delayed_healing": "No",
    "partial_paresis": "No",
    "muscle_stiffness": "No",
    "alopecia": "No",
    "obesity": "No"
}

try:
    response1 = requests.post("http://127.0.0.1:8000/predict",json=payload,timeout=10)

    response1.raise_for_status()

    data1 = response1.json()

    print("Predicted class:", data1["predicted_class"])
    print("Disclaimer:", data1["disclaimer"])

except requests.exceptions.Timeout:
    print("The request timed out.")

except requests.exceptions.ConnectionError:
    print("Could not connect. Check that the API server is running.")

except requests.exceptions.HTTPError as error:
    status = error.response

    if status == 422:
        print("The API rejected the input. Check the required fields and values.")
    else:
        print(f"The API returned an error. HTTP status: {status}")

except requests.exceptions.JSONDecodeError:
    print("The server response was not valid JSON.")

except requests.exceptions.RequestException:
    print("Another request error occurred.")
