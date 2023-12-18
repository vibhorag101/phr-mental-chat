import requests
import json
import pandas as pd

url = 'http://192.168.3.74:5000/predict_sentiment'
url2 = 'http://192.168.3.74:5000/predict_sentiment_scores'
def analyse_sentiment(text):
    data = {'text': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        emotions = response.json().get('emotions', [])
        return(emotions)
    else:
        print(f"Request failed with status code: {response.status_code}")

def analyse_sentiment_scores(text):
    data = {'text': text}
    response = requests.post(url2, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        emotions = response.json().get('emotions', [])
        return(emotions)
    else:
        print(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    print(analyse_sentiment("I am feeling very sad today."))
    print(analyse_sentiment_scores("Are you a fool?"))
