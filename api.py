from flask import Flask,request,jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/api',methods=['GET'])
def returnPred():
    d = {}
    
    inp  = str(request.args['query'])
    
    w = [[float(x) for x in inp.split(",")]]
    
    
    model = joblib.load('lib\kerala\model_loaded.joblib')
    

    
    d['output'] =str(model.predict(w))
    return d

if __name__ =='__main__':
    app.run()