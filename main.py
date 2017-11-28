import pandas as pd
from sklearn.preprocessing import scale
from flask import Flask, request
import forecast
import predictor as predictor

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
	age = request.form['age']
	workclass =  request.form['workclass']
	fnlwgt = request.form['fnlwgt']
	education = request.form['education']
	education_num = request.form['education_num']
	marital_status = request.form['marital_status']
	occupation = request.form['occupation']
	relationship =  request.form['relationship']
	race =  request.form['race']
	sex = request.form['sex']
	capital_gain = request.form['capital_gain']
	capital_loss = request.form['capital_loss']
	hours_per_week = request.form['hours_per_week']
	native_country = request.form['native_country']

	raw_user_input = {"age": age, "workclass":workclass, "fnlwgt":fnlwgt, "education":education, "education_num":education_num, "marital_status":marital_status, "occupation":occupation, "relationship":relationship, "race":race, "sex":sex, "capital_gain":capital_gain, "capital_loss":capital_loss, "hours_per_week":hours_per_week, "native_country":native_country}
	processed_user_input = predictor.predictClass(raw_user_input)
	
	return processed_user_input
	"""type age/workclass/fnlwgt etc to get the value from those attributes inputted by user in web"""	

if __name__ == '__main__':
	app.run()
