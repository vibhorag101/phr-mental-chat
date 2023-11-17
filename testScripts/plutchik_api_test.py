import requests
import json
import pandas as pd

url = 'http://192.168.3.74:5000/predict_sentiment'
def analyse_sentiment(text):
    data = {'text': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        result = response.json()
        return(parse_response(result))
    else:
        print(f"Request failed with status code: {response.status_code}")

def parse_response(result):
    data_dict = result["Your Data"][0]
    keys = list(data_dict.keys())[0].split(",")
    value = list(data_dict.values())[0].split(",")
    predicted_sentiment = [keys[i].split(" ")[0] for i in range(len(keys)) if value[i] == "1.0"]
    return predicted_sentiment

if __name__ == "__main__":
    print(analyse_sentiment("I am feeling very sad today."))
    print(analyse_sentiment("I lost so much money in a fire at my factory."))
    print(analyse_sentiment("My son is so caring."))
