import pandas as pd
import streamlit as st
import numpy as np
import requests
import pickle
from streamlit_lottie import st_lottie
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

col1, col2, col3 = st.columns(3)
with col2:
    st.title('Bankruptcy?')

st.sidebar.subheader("Our Mentor:")
st.sidebar.info("[Neha Ramchandani](https://www.linkedin.com/in/neha-p-a0b36a71/)")
st.sidebar.subheader("Group Members:")
st.sidebar.info("[Omkar Bhagwat](https://www.linkedin.com/in/omkar-bhagwat-64b103230/)")
st.sidebar.info("[Chaitravi Angane](https://www.linkedin.com/in/chaitravi-angane-556b29241/)")
st.sidebar.info("[Yogita Sharma](https://www.linkedin.com/in/yogita-mishra-8487b5161/)")
st.sidebar.info("[Amrut Vishwaroop](https://www.linkedin.com/in/amrut-vishwaroop-0ab946232/)")
st.sidebar.info("[Akarsh Bhasi](https://www.linkedin.com/in/akarshbhasi/)")
st.sidebar.info("[Manju Kiran]()")
st.sidebar.markdown("""
   <span style='color:orange;'>
   Made by India 🇮🇳</span>""",
   unsafe_allow_html=True)

with col2:
    def user_input_features():
        industrial_risk=col2.radio('Industrial Risk',[0, 0.5, 1],horizontal=True)
        operating_risk=col2.radio("Operating Risk",[0, 0.5, 1],horizontal=True)
        management_risk=col2.radio('Management Risk',[0, 0.5, 1],horizontal=True)
        financial_flexibility=col2.radio('Financial Flexibility',[0, 0.5, 1],horizontal=True)
        credibility=col2.radio("Credibility",[0, 0.5, 1],horizontal=True)
        competitiveness =col2.radio("Competitiveness",[0, 0.5, 1],horizontal=True)

        data={'Industrial Risk':industrial_risk,
              'Management Risk':management_risk,
              'financial_flexibility':financial_flexibility,
              'credibility':credibility,
              'competitiveness':competitiveness,
              'operating_risk':operating_risk}
    
        feature=pd.DataFrame(data,index = [0])
        return feature

def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_= load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_xeicuecf.json")
with col2:
    st_lottie(lottie_,key="emojis",height=100,width=200)
    
df=user_input_features()

data=pd.read_excel("/Users/om/Desktop/bankruptcy/Bankruptcy.xlsx")
labelencoder = LabelEncoder()
data.iloc[:, -1] = labelencoder.fit_transform(data.iloc[:,-1])

num_folds = 20
seed = 7
kfold = KFold(n_splits=num_folds)

array = data.values
X = array[:,0:6]
Y = array[:,6]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y,random_state = 42,
                                                   test_size = 0.25)
clf = GaussianNB()
clf.fit(Xtrain, ytrain)
Y_pred = clf.predict(Xtest)
acc_gaussian = round(clf.score(Xtrain, ytrain)*100,2)

with col2:
    prediction=clf.predict(df)
    st.subheader('Result')
if prediction==1:
    pred='It will be Not-Bankrupt'
else:
    pred='It will be Bankrupt'

with col2:
    if st.button("Predict"):
        st.success(pred)
        st.button("Clear output")
