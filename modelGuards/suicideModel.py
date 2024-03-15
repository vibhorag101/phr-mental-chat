from transformers import pipeline
classifier = pipeline(task="text-classification", model="vibhorag101/roberta-base-suicide-prediction-phr",device_map="auto")
# can classify 2 emotions (suicidal, non-suicidal)
def predict_suicide(text):
    emotions = classifier(text)
    return(emotions[0]['label'])

if __name__ == "__main__":
    print(predict_suicide("I am worthless"))
    print(predict_suicide("I am a happy man"))