import pickle 
import streamlit as st 


pickle_in = open('Customer_Churn_pred_gb.pkl', 'rb') 
classifier = pickle.load(pickle_in) 
  
def welcome(): 
    return 'welcome all'

def prediction(CreditScore, Geography, Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):   
   
    prediction = classifier.predict( 
        [[CreditScore, Geography, Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]) 
    print(prediction) 
    return prediction

def main():  
    st.title("Customer Churn Prediction") 
      
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Customer Churn Classifier ML App </h1> 
    </div> 
    """
 
    st.markdown(html_temp, unsafe_allow_html = True) 

    CreditScore = st.slider("Credit Score", 100,1000, step =1) 
    Geography_select = st.selectbox("Geography", [None]+["Spain","France","Germany"])   # Spain =2 , France = 0 , Germany =1
    if Geography_select == "Spain":
        Geography = 2
    elif Geography_select == "France":
        Geography = 0
    else:
        Geography = 1        
    Gender_select = st.selectbox("Gender", [None]+["Male","Female"])  #Male =1 , Female =0
    if Gender_select == "Male":
        Gender = 1
    else:
        Gender = 0    
    Age = st.number_input("Age") 
    Tenure = st.slider("Tenure", 0 ,10,step=1)
    Balance = st.number_input("Balance")
    NumOfProducts = st.slider("Number of Products", 1 ,4,step=1)
    credit_card = st.selectbox("Has credit card?", [None]+["Yes","No"])
    if credit_card == "Yes":
        HasCrCard = 1
    else:
        HasCrCard = 0
    Active_Member = st.selectbox("Is Active Member?", [None]+["Yes","No"])
    if Active_Member == "Yes":
        IsActiveMember = 1
    else:
        IsActiveMember = 0
    EstimatedSalary = st.number_input("Estimated Salary")
    result =""  
    if st.button("Predict"): 
        result = prediction(CreditScore, Geography, Gender, Age,Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        if result == 1:
            result = " has Exited the Bank"
        else:
            result = "is With the Bank"

    st.success('The Customer {}'.format(result)) 
     
if __name__=='__main__': 
    main() 
