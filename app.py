from flask import Flask,render_template,url_for,redirect,session
import numpy as np
from wtforms import  FloatField,SubmitField
from wtforms.validators import InputRequired,NumberRange
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
import pandas as pd
import joblib

df=pd.read_csv("iris.csv")


def prediction(model,sample_json):
	s_len=sample_json['sepal_length']
	s_wid=sample_json['sepal_width']
	p_len=sample_json['petal_length']
	p_wid=sample_json['petal_width']
	flower = np.array([s_len,s_wid,p_len,p_wid])
	y_pred=model.predict(flower.reshape(1,4))
	print(y_pred)
	return y_pred


app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'mykey'


class IrisForm(FlaskForm):
	sep_len=FloatField('Sepal Length',validators=[InputRequired(message="Data is Required"),NumberRange(message="Must be in "+str(df['sepal_length'].min())+" To "+str(df['sepal_length'].max()),min=df['sepal_length'].min(),max=df['sepal_length'].max())])
	sep_wid=FloatField('Sepal Width',validators=[InputRequired(message="Data is Required"),NumberRange(message="Must be in "+str(df['sepal_width'].min())+" To "+str(df['sepal_width'].max()),min=df['sepal_width'].min(),max=df['sepal_width'].max())])
	pet_len=FloatField('Petal Length',validators=[InputRequired(message="Data is Required"),NumberRange(message="Must be in "+str(df['petal_length'].min())+" To "+str(df['petal_length'].max()),min=df['petal_length'].min(),max=df['petal_length'].max())])
	pet_wid=FloatField('Petal Width',validators=[InputRequired(message="Data is Required"),NumberRange(message="Must be in "+str(df['petal_width'].min())+" To "+str(df['petal_width'].max()),min=df['petal_width'].min(),max=df['petal_width'].max())])

	submit = SubmitField('Analyze')



@app.route('/',methods=['GET','POST'])
def index():
	form=IrisForm()
	if form.validate_on_submit():
		session['sep_len']=form.sep_len.data
		session['sep_wid']=form.sep_wid.data
		session['pet_len']=form.pet_len.data
		session['pet_wid']=form.pet_wid.data

		return redirect(url_for("flower_prediction"))
	return render_template('home.html',form=form)


model = joblib.load('iris.pkl')

@app.route('/prediction')
def flower_prediction():
	content = {}
	content["sepal_length"] = session['sep_len']
	content["sepal_width"] = session['sep_wid']
	content["petal_length"] = session['pet_len']
	content["petal_width"] = session['pet_wid']

	results =prediction(model,content)

	return render_template("prediction.html",results=results)


if __name__ == '__main__':
	app.run(debug=True)