from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# Routes

@app.route('/', methods=['GET'])  
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET']) 
def index():
    if request.method == 'POST':
        try:
            # Read inputs from form
            merchant = request.form['merchant']
            category = request.form['category']
            amt = float(request.form['amt'])
            lat = float(request.form['lat'])
            long = float(request.form['long'])
            city_pop = int(request.form['city_pop'])
            job = request.form['job']
            unix_time = int(request.form['unix_time'])
            merch_lat = float(request.form['merch_lat'])
            merch_long = float(request.form['merch_long'])
            trans_hour = int(request.form['trans_hour'])
            trans_day = int(request.form['trans_day'])
            trans_month = int(request.form['trans_month'])
        
            # Create DataFrame with user input
            data_df = pd.DataFrame({
                "merchant": [merchant],
                "category": [category],
                "amt": [amt],
                "lat": [lat],
                "long": [long],
                "city_pop": [city_pop],
                "job": [job],
                "unix_time": [unix_time],
                "merch_lat": [merch_lat],
                "merch_long": [merch_long],
                "trans_hour": [trans_hour],
                "trans_day": [trans_day],
                "trans_month": [trans_month]
            })

            # Prediction
            obj = PredictionPipeline()
            predict = obj.predict(data_df)
            predicted_result = "Suspicious Transaction!" if predict == 1 else "Normal Transaction"

            return render_template('results.html', prediction=predicted_result)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong!'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
