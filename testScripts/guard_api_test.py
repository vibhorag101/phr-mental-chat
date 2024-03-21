import requests
import json

url = 'http://192.168.3.74:6006/safety'
def analyze_suicide_sentiment(text):
    data = {'message': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        result = response.json()
        return(result)
    else:
        return(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    print(analyze_suicide_sentiment("I am sad today"))