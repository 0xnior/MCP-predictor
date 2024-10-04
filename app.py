from flask import Flask, render_template, redirect, request,jsonify
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open("PycharmProjects/pythonProject1/IEX_model.pickle", 'rb'))
scaler = pickle.load(open("scaler.pickle", 'rb'))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home_page():

    return render_template('index.html',
                           prediction=-1)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json  # Expecting JSON format
    df = pd.DataFrame(data, index=[0])  # Convert to DataFrame
    return jsonify(prediction=pipe.predict(scaler.transform(df)).tolist())  # Return the prediction as JSON


if __name__ == '__main__':
    app.run(debug=True)
