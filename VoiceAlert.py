#!/bin/sh
"exec" "`dirname $0`/venv/bin/python" "$0" "$@"
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
import sys
import pyttsx3
import webbrowser
import random
import threading ,time
import pyglet


CONFIG={'intents':{'alert':{'example':['помогите','стой','остановись','больно','вытяни','помоги','мы все умрем','отпусти','aaaaaaaaa','страдание',
                                       'боль','не двигайся','замолчи'],
                 'answer':'открыть дверь'},
        'fine':{'example':['привет','как дела','куда едешь','как учеба','как семья','кочешь есть','погода','какие планы на вечер','хорошо выглядишь',
                           'хочешь выпить','касивое облако'],
                 'answer':'едем дальше'}}}
dataset=[]
for intent,intent_value in CONFIG['intents'].items():
    for example in intent_value['example']:
        dataset.append([example,intent])
#print(dataset)
corpus = [example for example,answer in dataset]
y = [answer for example,answer in dataset]
#print(corpus)

vectorize = CountVectorizer()
X = vectorize.fit_transform(corpus)
#print(X)
clf =LogisticRegression()
clf.fit(X,y)

def predict(X):
    return clf.predict(vectorize.transform([X]))
def cmd():
    r=sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        print("Говорите")
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
    try:
        task=r.recognize_google(audio,language="ru-RU").lower()
        print(task)
        res= predict(task)
        print(res)
    except sr.UnknownValueError:
        print('Я вас не понял')
        task= cmd()
    return task


while True:
    cmd()
