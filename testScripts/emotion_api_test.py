import requests
import json
from generation_api_test import sendRequest

# Just to show usage of the emotion API.
# Although can just import the model from emotion_classifier.py in the application.
url = 'http://192.168.3.74:5555/emotion'
def analyse_sentiment(text):
    data = {'message': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    # if response.status_code == 200:
    result = response.json()
    return(result)
    # else:
        # print(f"Request failed with status code: {response.status_code}")

def analyse_sentiment_model(text):
    print("Query: "+text)
    response = sendRequest(text)
    print("Response: "+response)
    print("Emotion in response: "+analyse_sentiment(response))

if __name__ == "__main__":
    print(analyse_sentiment("I am feeling very sad today."))
    print(analyse_sentiment("I lost so much money in a fire at my factory."))
    print(analyse_sentiment("My son is so caring."))