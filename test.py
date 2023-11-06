import requests

# Define the URL of your FastAPI application
url = "http://localhost:8000/generate"

# Prepare the initial request data
data = {
    "message": "I am feeling very sad today.",
    "history": [],
    "system_prompt": "Test prompt",
    "max_new_tokens": 500,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 25
}

def sendRequest(message):
    data["message"] = message
    response = requests.post(url, json=data)
    response_body = response.json()
    history = [tuple(lst) for lst in response_body]
    data["history"] = history
    print(history)

sendRequest("I am feeling very sad today.")
sendRequest("I lost so much money in a fire at my factory.")
