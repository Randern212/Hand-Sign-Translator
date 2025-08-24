import cv2
import tensorflow as tf
import numpy as np
import requests as rq

ipAddress:str="http://192.168.4.1/send" #TODO adjust to the ESP's IP

predictingNumber:bool = False
predictedClass=None
predictionCount=0
frameHoldThreshold=23 #TODO adjust this with trial and error 

numberModel=tf.keras.models.load_model("models\numberModel.keras")
letterModel=tf.keras.models.load_model("models\letterModel.keras")

numbers = [str(i) for i in range(1, 11)]
letters = [chr(i) for i in range(65, 91)] +["nothing", " "]

cap = cv2.VideoCapture(0)

def sendPrediction(prediction,classNames):
    classID=np.argmax(prediction)
    classString=classNames[classID]
    rq.get(ipAddress,params={"data":classString})

while cap.isOpened():
    model=letterModel 
    labels=letters
    if predictingNumber:
        model=numberModel
        labels=numbers
    ret, frame = cap.read() 
    if not ret:
        break  
    frame = cv2.flip(frame, 1) 
    
    cv2.imshow('Sign Language Translator', frame)
    
    img=cv2.resize(frame, (200,200))
    img=img.astype("float32")
    img=np.expand_dims(img, axis=0)

    pred=model.predict(img)

    if pred != predictedClass:
        predictedClass=pred
        predictionCount=0
    else:
        predictionCount+=1

    if predictionCount>=frameHoldThreshold:
        sendPrediction(predictedClass,labels)
        predictionCount=0

    if cv2.waitKey(5) & 0xFF == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
