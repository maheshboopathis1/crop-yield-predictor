import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('cropdata.csv')



dataset_X = dataset.iloc[:,[0,1,2,3,4,5,6]].values
dataset_Y = dataset.iloc[:,7].values


dataset_X


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)



dataset_scaled = pd.DataFrame(dataset_scaled)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    n=int(request.form['N'])
    p =int(request.form['P'])
    k = int(request.form['K'])
    t=float(request.form['Temperature'])
    h=float(request.form['Humidity'])
    ph=float(request.form['PH'])
    r=float(request.form['Rainfall'])
    data = np.array([[n,p,k,t,h,ph,r]])

    if(n<240):
        return render_template('index.html', msg_text='{}'.format('nitrogen(N) value must be greater than 240'))
    if(n>480):
        return render_template('index.html', msg_text='{}'.format('nitrogen(N) value must be less than than 480'))
    if(p<11):
        return render_template('index.html', msg_text='{}'.format('phosphorus(P) value must be greater than 11'))
    if(p>22):
        return render_template('index.html', msg_text='{}'.format('phosphorus(P) value must be less than than 22'))
    if(k<110):
        return render_template('index.html', msg_text='{}'.format('potassium(K) value must be greater than 110'))
    if(k>280):
        return render_template('index.html', msg_text='{}'.format('potassium(K) value must be less than than 280'))
   
    prediction = model.predict( sc.transform(data) )
    print(prediction)
   
    output = prediction

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
