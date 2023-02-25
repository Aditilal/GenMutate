import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template("index.html")



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        ID = int(request.form['ID'])
        GENE=request.form['GENE']
        VARIATION=request.form['VARIATION']
        CLASS=request.form['CLASS']
        form_Array = np.array([[ID,GENE,VARIATION,CLASS]])
        model = pickle.load(open('geneticmutation.pkl', 'rb'))
        # prediction= model.predict(form_Array)
        
        
        result = "Silent mutations cause a change in the sequence of bases in a DNA molecule, but do not result in a change in the amino acid sequence of a protein"
        image = "class7.png"
    
    return render_template("result.html",result = result,image = image)

if __name__=="__main__":
    app.run(debug=True)