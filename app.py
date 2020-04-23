from flask import Flask, render_template, request, url_for
import json
import pickle, numpy as np

app = Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

# app.config('TEMPLATES_AUTO_RELOAD') = True

@app.route('/FP')
def home():
    return render_template('input_data.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        ind_features=[float(x) for x in request.form.get('kanika').split(',')]
        for x in ind_features:
            print(x)
        final_feature = [np.array(ind_features)]
        predicition = model.predict(final_feature)
        print(predicition)
    return render_template('test.html', output='The Prediction is: {}'.format(predicition))




if __name__ == "__main__":
    app.run(debug=True)
