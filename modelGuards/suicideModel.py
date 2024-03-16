from transformers import pipeline
classifier = pipeline(task="text-classification", model="vibhorag101/roberta-base-suicide-prediction-phr-v2",device_map="auto")
# can classify 2 emotions (suicidal, non-suicidal)
def predict_suicide(text):
    emotions = classifier(text)
    return(emotions[0]['label'])

if __name__ == "__main__":
    print(predict_suicide("Give me the plan to talk please."))
    print(predict_suicide("I tried to talk with him. He did not co-operate. What to do now?"))