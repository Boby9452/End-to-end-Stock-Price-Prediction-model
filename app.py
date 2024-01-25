import numpy as np
import pandas as pandas
from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
sclr= StandardScaler()

# Loading model
model=pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('sclr.pkl', 'rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    Open=request.form['Open']
    High=request.form['High']
    Low=request.form['Low']
    Adj_Close=request.form['Adj_Close']
    Volume=request.form['Volume']
    year=request.form['year']
    month=request.form['month']
    day=request.form['day']

    features=np.array([[Open,High,Low,Adj_Close,Volume,year,month,day]])
    feature_scaled=scaler.transform(features)
    prediction=model.predict(feature_scaled).reshape(1,-1)

    return render_template('index.html',output=prediction[0])

if __name__=="__main__":
    app.run(debug=True)


