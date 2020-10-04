from flask import Flask,render_template,request
import numpy as np
import pickle

model=pickle.load(open('heart_dis.pickle','rb'))



app=Flask(__name__)
@app.route('/')
def home():

    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


#([[age,sex,cp,trestbps,col,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

    return render_template('index.html',pre_text="predicted : {}".format(prediction[0]))


if __name__=='__main__':
    app.run(debug=True)
