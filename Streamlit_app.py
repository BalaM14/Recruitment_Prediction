import numpy as np
import pickle
import pandas as pd
import streamlit as st 

"""
### Created By : Bala Murugan
#### LinkedIn : https://www.linkedin.com/in/balamurugan14/

# Recruitment Hiring Prediction
"""

pickle_in = open("model.pkl","rb")
regressor=pickle.load(pickle_in)


def main():
    html_temp = """
    <div style="background-color:slateblue;padding:10px">
    <h2 style="color:black;text-align:center;">Recruitment Hiring Predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age","")
    ExperienceYears = st.text_input("ExperienceYears","")
    PreviousCompanies = st.text_input("PreviousCompanies","")
    DistanceFromCompany = st.text_input("DistanceFromCompany","")
    InterviewScore = st.text_input("InterviewScore","")
    SkillScore = st.text_input("SkillScore","")
    PersonalityScore = st.text_input("PersonalityScore","")
    Gender = st.text_input("Gender","")
    EducationLevel = st.text_input("EducationLevel","")
    RecruitmentStrategy = st.text_input("RecruitmentStrategy","")
    result=""

    input_data = {"Age": [Age], "ExperienceYears": [ExperienceYears], "PreviousCompanies": [PreviousCompanies],
                  "DistanceFromCompany": [DistanceFromCompany], "InterviewScore": [InterviewScore], 
                  "SkillScore":[SkillScore],"PersonalityScore": [PersonalityScore], 
                  "Gender":[Gender], "EducationLevel": [EducationLevel],
                   "RecruitmentStrategy": [RecruitmentStrategy]}
    
    dataframe = pd.DataFrame(input_data)

    if st.button("Predict Recruitment Status"):
        result=regressor.predict(dataframe)
    if result == 1:
        st.success('The output is Candidate will be Hired')
    else:
        st.error('The output is Candidate will not be Hired')

    st.write("")
    st.write(f"{'=='*17}   BATCH PREDICTION   {'=='*18}")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Input Features: limited to first 5 rows")
        st.dataframe(data.head())

        if st.button("Run"):
            data['Predicted_Price'] = regressor.predict(data)
            st.write("Output Predicted Features: limited to first 5 rows")
            st.dataframe(data.head())

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')


            csv = convert_df(data)

            st.download_button(
            "Download Predicted Output File",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
            )

if __name__=='__main__':
    main()