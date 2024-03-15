from detoxify import Detoxify

#https://github.com/unitaryai/detoxify/

toxicModel = Detoxify('original')

def predictThreat(text):
    res = toxicModel.predict(text)
    # print(res)
    threatList=[]
    for key in res:
        if(res[key]>0.5):
            threatList.append(key)
    if(len(threatList)!=0):
        return("threat")
    else:
        return("safe")
    
if __name__ == "__main__":
    print(predictThreat("I dont wish to live anymore"))