import pandas as pd
import numpy as np
from flask import Flask,request
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
pickle_in = open('random_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def Welcome():
    return "Welcome To The Home Page"

@app.route('/predict', methods=["GET"])
def predict_note_authentication():
    """Let's Authenticate the Banks Note
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Output Values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The Predicted Class is" + str(prediction)


if __name__=="__main__":
    app.run()