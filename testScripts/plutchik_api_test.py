import requests
import json
import pandas as pd

url = 'http://192.168.3.74:5000/predict_sentiment'
def analyse_sentiment(text):
    data = {'text': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        emotions = response.json().get('columns', [])
        return(emotions)
    else:
        print(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    print(analyse_sentiment("I am feeling very sad today."))
    print(analyse_sentiment("I lost so much money in a fire at my factory."))
    print(analyse_sentiment("In the tender embrace of shared moments, their souls danced in unity, weaving a tapestry of emotions that transcended words, creating a profound connection that resonated in the silent language of the heart."))
