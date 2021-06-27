from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np



import re

app = Flask(__name__)
#heartdiseases model read
filename = open('HeartDiseases/heartdiseasespredictmodel.pkl', 'rb')
clf = pickle.load(filename)
filename.close()

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/hdpredict', methods=['GET','POST'])
def hdpredict():
    if request.method == 'POST':
        na = request.form['na']
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = float(request.form['exang'])
        slope = float(request.form['slope'])
        ca =   float(request.form['ca'])
        thal =  float(request.form['thal'])
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,slope,ca,thal]])
        my_prediction = clf.predict(data)
        my_prediction_proba = clf.predict_proba(data)[0][1]
        
        return render_template('hdpshow.html',name=na,prediction=my_prediction,proba=my_prediction_proba)
    return render_template('hdp.html')

        
if __name__ == '__main__':
	app.run(debug=True)