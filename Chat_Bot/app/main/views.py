from django.shortcuts import render
# import requests
# from .chatbot import *
# from .apps import WebappConfig
import random
import json
import pickle
from django.http import HttpResponse
import numpy as np
import requests
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from django.shortcuts import render
from django.urls import reverse

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('/home/himanshu/celetel/Chat_Bot/app/main/ML_files/intents.json').read())

words = pickle.load(open('/home/himanshu/celetel/Chat_Bot/app/main/words.pkl', 'rb'))
classes = pickle.load(open('/home/himanshu/celetel/Chat_Bot/app/main/classes.pkl', 'rb'))
model = load_model('/home/himanshu/celetel/Chat_Bot/app/main/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is Running")

def main(message):
    message = message
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    return res

# Create your views here.
def index(request):
    if request.method == "POST":
        data = request.POST["data"]
        # result = main(data)
        # print(result)
        return render(request, "home.html",{"message": data})
    
    return render(request, "home.html",{"message": ""})
    