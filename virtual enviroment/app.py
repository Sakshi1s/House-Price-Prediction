from flask import Flask, render_template, request, jsonify ,redirect
import json
import requests
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the trained model
pickled_model = pickle.load(open('gbr.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

train = pd.read_csv('housing_train.csv')

string_features = ['Neighborhood', 'Condition1', 'BldgType']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(train[string_features])
numerical_features = ['YearBuilt','BedroomAbvGr', 'KitchenAbvGr','FullBath', 'GarageCars', 'LotArea']
features = np.concatenate((encoded_features, train[numerical_features]), axis=1)
target = train['SalePrice']
gbr = GradientBoostingRegressor()
gbr.fit(features, target)


@app.route('/')

def home():
    return render_template('index.html')
@app.route('/open_second_page', methods=['POST'])

def open_second_page(): 
    return redirect('/second_page')

@app.route('/second_page')
def second_page():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    Neighborhood = request.form['Neighborhood']
    Condition1 = request.form['Condition1']
    year_built = int(request.form['YearBuilt'])
    BldgType=request.form['BldgType']
    BedroomAbvGr = int(request.form['BedroomAbvGr'])
    KitchenAbvGr= int(request.form['KitchenAbvGr'])
    FullBath= int(request.form['FullBath'])
    GarageCars= int(request.form['GarageCars'])
    LotArea= int(request.form['LotArea'])

    # Encode the new instance
    encoded_instance = encoder.transform([[Neighborhood,Condition1,BldgType]])
    
    # Combine encoded instance with numerical features
    instance_features = pd.concat([pd.DataFrame(encoded_instance), pd.DataFrame([[year_built,BedroomAbvGr,KitchenAbvGr,FullBath,GarageCars,LotArea]])], axis=1)
    
    # Make predictions
    predicted_prices = gbr.predict(instance_features)[0]
    
    return render_template('result.html', predicted_price=predicted_prices)


if __name__ == '__main__':
    app.run(debug=True)
