import pandas as pd
import numpy as np
from flask import Flask,request
import pickle

app = Flask(__name__)
pickle_in = open('random_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def Welcome():
    return "Welcome To The Home Page"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Predicted Class is"+ str(prediction)

if __name__=="__main__":
    app.run()