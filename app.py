from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__Home Price Predictor__)


model_s = joblib.load('trained_model.pkl')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features = [
        float(request.form['median_income']),
        float(request.form['housing_median_age']),
        float(request.form['total_rooms']),
        float(request.form['total_bedrooms']),
        float(request.form['population']),
        float(request.form['households']),
        float(request.form['median_house_value']),
        request.form['ocean_proximity']
    ]
    
    
    ocean_proximity_encoded = encoder.transform([features[-1]])
    
    
    features_encoded = list(features[:-1]) + list(ocean_proximity_encoded.toarray()[0])
    
    
    
    prediction = model_s.predict(features_scaled)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)