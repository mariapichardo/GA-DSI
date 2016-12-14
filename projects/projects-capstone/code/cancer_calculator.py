import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pickle


with open("lr.pkl", "r") as f:
    PREDICTOR = pickle.load(f)

with open('X_age_menarche_dum.pkl', 'r') as f:
    X_age_menarche_dum  = pickle.load(f)
    
with open('X_age_group_dum.pkl', 'r') as f:
    X_age_group_dum  = pickle.load(f)

with open('X_race_dum.pkl', 'r') as f:
    X_race_dum  = pickle.load(f)
    
with open('X_hist_fam_dum.pkl', 'r') as f:
    X_hist_fam_dum  = pickle.load(f)
    
with open('X_hist_self_dum.pkl', 'r') as f:
    X_hist_self_dum   = pickle.load(f) 
    
with open('X_age_f_birth_dum.pkl', 'r') as f:
    X_age_f_birth_dum  = pickle.load(f)
    
with open('X_horm_repl_dum.pkl', 'r') as f:
    X_horm_repl_dum  = pickle.load(f)

with open('X_menop_dum.pkl', 'r') as f:
    X_menop_dum  = pickle.load(f)
    
with open('X_bmi_dum.pkl', 'r') as f:
   X_bmi_dum  = pickle.load(f)        

#-------- FUNCTION GOES HERE -----------#

def dummy_age_group(age_group_5_years):
    for i, j in enumerate(X_age_group_dum.index):
        if str(age_group_5_years) == j.split("_") [1]:
            X_age_group_dum.iloc[i] = 1
        else: 
            X_age_group_dum.iloc[i] = 0
    return X_age_group_dum   

def dummy_age_menarche(age_menarche):
    for i, j in enumerate(X_age_menarche_dum.index):
        if str(age_menarche) == j.split("_") [1]:
            X_age_menarche_dum.iloc[i] = 1
        else: 
            X_age_menarche_dum.iloc[i] = 0
    return X_age_menarche_dum  

def dummy_race_eth(race_eth):
    for i, j in enumerate(X_race_dum.index):
        if str(race_eth) == j.split("_") [1]:
            X_race_dum.iloc[i] = 1
        else: 
            X_race_dum.iloc[i] = 0
    return X_race_dum  


def dummy_first_degree_hx(first_degree_hx):
    for i, j in enumerate(X_hist_fam_dum.index):
        if str(first_degree_hx) == j.split("_") [1]:
            X_hist_fam_dum.iloc[i] = 1
        else: 
            X_hist_fam_dum.iloc[i] = 0
    return X_hist_fam_dum  

def dummy_breast_cancer_history(breast_cancer_history):
    for i, j in enumerate(X_hist_self_dum.index):
        if str(breast_cancer_history) == j.split("_") [1]:
            X_hist_self_dum.iloc[i] = 1
        else: 
            X_hist_self_dum.iloc[i] = 0
    return X_hist_self_dum  

def dummy_menopaus(menopaus):
    for i, j in enumerate(X_menop_dum.index):
        if str(menopaus) == j.split("_") [1]:
            X_menop_dum.iloc[i] = 1
        else: 
            X_menop_dum.iloc[i] = 0
    return X_menop_dum   


def dummy_bmi_group(bmi_group):
    for i, j in enumerate(X_bmi_dum.index):
        if str(bmi_group) == j.split("_") [1]:
            X_bmi_dum.iloc[i] = 1
        else: 
            X_bmi_dum.iloc[i] = 0
    return X_bmi_dum   
    
def dummy_age_first_birth(age_first_birth):
    for i, j in enumerate(X_age_f_birth_dum.index):
        if str(age_first_birth) == j.split("_") [1]:
            X_age_f_birth_dum.iloc[i] = 1
        else: 
            X_age_f_birth_dum.iloc[i] = 0
    return X_age_f_birth_dum  

def dummy_current_hrt(current_hrt):
    for i, j in enumerate(X_horm_repl_dum.index):
        if str(current_hrt) == j.split("_") [1]:
            X_horm_repl_dum.iloc[i] = 1
        else: 
            X_horm_repl_dum.iloc[i] = 0
    return X_horm_repl_dum                 
    

#-------- ROUTES GO HERE -----------#

# This method takes input via an HTML page
@app.route('/page')
def page():
   with open("page_breast_cancer.html", 'r') as viz_file:
       return viz_file.read()

# @app.route("/predict", methods=["GET"])
# def predict():
#     pclass = flask.request.args['pclass']
#     sex = flask.request.args['sex']
#     age = flask.request.args['age']
#     fare = flask.request.args['fare']
#     sibsp = flask.request.args['sibsp']

#     item = [pclass, sex, age, fare, sibsp]
#     score = PREDICTOR.predict_proba(item)
#     results = {'survival chances': score[0,1], 'death chances': score[0,0]}
#     return flask.jsonify(results)

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

      
       X_age_group_dum = dummy_age_group(inputs['age_group_5_years'])
       X_age_menarche_dum  = dummy_age_menarche(inputs['age_menarche'])
       X_race_dum  = dummy_race_eth(inputs['race_eth'])
       X_hist_fam_dum  = dummy_first_degree_hx(inputs['first_degree_hx'])
       X_hist_self_dum   = dummy_breast_cancer_history(inputs['breast_cancer_history'])
       X_age_f_birth_dum  = dummy_age_first_birth(inputs['age_first_birth'])
       X_horm_repl_dum  = dummy_current_hrt(inputs['current_hrt'])
       X_menop_dum  = dummy_menopaus(inputs['menopaus'])
       X_bmi_dum  = dummy_bmi_group(inputs['bmi_group'])

       item = pd.concat([X_age_menarche_dum, X_age_group_dum, X_race_dum, X_hist_fam_dum, X_hist_self_dum , X_age_f_birth_dum , X_horm_repl_dum, X_menop_dum, X_bmi_dum 

])

      
       score = PREDICTOR.predict_proba(item)
       results = {'No Breast Cancer Prob': score[0,1], 'Yes Breast Cancer Prob': score[0,0]}
       return flask.jsonify(results) 

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT)