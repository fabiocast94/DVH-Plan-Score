import streamlit as st
import pandas as pd

st.title("Plan Quality Dashboard")

uploaded_file = st.file_uploader("Carica CSV del piano (DVH / metriche)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dati caricati:")
    st.dataframe(df.head())
    
    ptv_coverage = df['PTV_Coverage'].mean()
    oar1_dose = df['OAR1_MeanDose'].mean()
    oar2_dose = df['OAR2_MeanDose'].mean()
    
    score = 0.5*ptv_coverage + 0.25*(50 - oar1_dose) + 0.25*(50 - oar2_dose)
    st.write(f"**Plan Score:** {score:.2f}")
    st.line_chart(df[['PTV_Coverage','OAR1_MeanDose','OAR2_MeanDose']])
