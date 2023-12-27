import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd 
app =Flask(__name__)
model=pickle.load(open('regression_model.pkl','rb'))
scalar=pickle.load(open('scaler_model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(np.array(list(data.values())))
    new_data=scalar.transform(np.array(list(data.values())))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x  in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The house price is{}.format(output)")
    

if __name__=="__main__":
    app.run(debug=True)