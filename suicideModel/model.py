from transformers import pipeline

# Using https://huggingface.co/SamLowe/roberta-base-go_emotions
# Can classify 28 emotions
def predict_suicide(text):
    classifier = pipeline(task="text-classification", model="vibhorag101/roberta-base-suicide-prediction-phr",device_map="auto")
    emotions = classifier(text)
    return(emotions[0]['label'])

if __name__ == "__main__":
    print(predict_suicide("I am worthless"))