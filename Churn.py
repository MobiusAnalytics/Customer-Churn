#%%writefile app.py

import pickle
import numpy as np
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from PIL import Image


# loading the trained model
pickle_file = open('RF_churn_model.sav', 'rb') 
Model = pickle.load(pickle_file)
titlepage = Image.open('titlepage.png')

@st.cache()

def prediction(Tenure,account_segment,Complain_ly,CC_Agent_Score,Marital_Status,Day_Since_CC_connect):
    if account_segment == "Regular":
        account_segment=1
    elif account_segment == "Regular Plus":
        account_segment=2
    elif account_segment == "Super":
        account_segment=3
    elif account_segment == "Super Plus":
        account_segment=4
    elif account_segment == "HNI":
        account_segment=5
        
    if Marital_Status == "Single":
        Marital_Status=1 
    elif Marital_Status == "Married":
        Marital_Status=2 
    elif Marital_Status == "Divorced":
        Marital_Status=3 
        
        
    if Complain_ly == "YES":
        Complain_ly =1
    elif Complain_ly == "NO":
        Complain_ly =0

    Predict = Model.predict([[Tenure,account_segment,Complain_ly,CC_Agent_Score,Marital_Status,Day_Since_CC_connect]])
    
    return Predict
    
def main(): 
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Campaign Performance Prediction</h1> 
    </div> 
    """    
    #st.markdown(html_temp, unsafe_allow_html = True) 
    #st.title("Churn Prediction")
    st.image(titlepage)
    st.sidebar.subheader("Upload a file to Predict the output!")
    uploaded_file = st.sidebar.file_uploader("Choose a File")
    if uploaded_file is not None:
    # To predict a test dataframe!!!
        dataframe = pd.read_csv(uploaded_file)
        dataframe = dataframe.replace({'account_segment' : { 'Regular' : 1, 'Regular Plus' : 2, 'Super' : 3,'Super Plus':4,'HNI':5}})
        dataframe = dataframe.replace({'Marital_Status' : { 'Single' : 2, 'Married' : 1, 'Divorced' : 3}})
        output = Model.predict(dataframe)
        #output = int(output)            
        dataframe['Response'] = output
        dataframe = dataframe.replace({'Response' : { 0 :'Not_Churn', 1:'Churn'}})
        dataframe = round(dataframe)        
        st.write(dataframe)
        dataframe =dataframe.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download CSV',data=dataframe,mime='text/csv',file_name='Download.csv')


    st.subheader("")
    st.subheader("Enter the values to predict churn")
    Tenure = st.number_input("Tenure of the account",min_value=0,max_value=80,step=1)
    account_segment = st.selectbox('Account Type',("Regular","Regular Plus","Super","Super Plus","HNI"))
    Complain_ly = st.selectbox('Did the Customer raised any Complaints last year?',("YES","NO"))
    CC_Agent_Score = st.selectbox('Customer care agent Feedback',(1,2,3,4,5))
    Marital_Status = st.selectbox('Marital_Status',("Single","Married","Divorced"))
    Day_Since_CC_connect = st.number_input("Enter the days since last Customer care connect",min_value=0,max_value=60,step=1)
    result = ""
    
    st.markdown("Predict the Customer status")
    if st.button("PREDICT"): 
        result = prediction(Tenure,account_segment,Complain_ly,CC_Agent_Score,Marital_Status,Day_Since_CC_connect)
        if result == 0:
            st.success("Based on the given attributes, Model predicts that the Customer will NOT CHURN")
        else:
            st.image("""https://www.customerthermometer.com/img/Blog-12.jpg""")
        
if __name__=='__main__': 
    main()
