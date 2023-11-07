import requests

# Define the URL of your FastAPI application
url = "http://192.168.3.74:5555/generate"
# Prepare the initial request data
data = {
    "message": "I am feeling very sad today.",
    "history": [],
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
