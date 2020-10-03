from flask import Flask,render_template,request
import numpy as np
import pickle
from sklearn.svm import SVC
model=pickle.load(open('heart_dis.pickle','rb'))



app=Flask(__name__)
@app.route('/')
def home():

    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():

    age=int(request.form["age"])
    sex=int(request.form["sex"])

    cp=int(request.form["cp"])
    trestbps=int(request.form["trestbps"])
    col=int(request.form["col"])
    fbs=int(request.form["fbs"])
    restecg=int(request.form["restecg"])
    thalach=int(request.form["thalach"])
    oldpeak=float(request.form["oldpeak"])
    slope=int(request.form["slope"])
    ca=int(request.form["ca"])
    thal=int(request.form["thal"])
    arr=np.array([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])
    model.predict(arr)


#([[age,sex,cp,trestbps,col,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

    return render_template('index.html',pre_text="predicted : {}".format([age,sex]))


if __name__=='__main__':
    app.run(debug=True)
