import requests
import json

# Just to show usage of the emotion API.
# Although can just import the model from emotion_classifier.py in the application.
url = 'http://192.168.3.74:5555/emotion'
def analyse_sentiment(text):
    data = {'message': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        result = response.json()
        return(result)
    else:
        print(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    print(analyse_sentiment("I have not eaten anything since 10 days. I am dying."))
    print(analyse_sentiment("I got a first rank in my class today."))