import streamlit as st
import pydicom
from dicompylercore import dicomparser, dvhcalc
import numpy as np
import plotly.express as px

st.title("Plan Quality Dashboard - RTDOSE/RTSTRUCT")

# Caricamento dei file DICOM
uploaded_struct = st.file_uploader("Carica RTSTRUCT", type=["dcm"])
uploaded_dose = st.file_uploader("Carica RTDOSE", type=["dcm"])

if uploaded_struct and uploaded_dose:
    try:
        # Lettura dei file DICOM
        struct = pydicom.dcmread(uploaded_struct)
        dose = pydicom.dcmread(uploaded_dose)

        # Parser del RTSTRUCT
        dp = dicomparser.DicomParser(struct)
        organs = dp.GetStructures()  # dict con ID: {Name, Type, Color}

        st.subheader("Organi disponibili nel piano")
        organ_list = [organs[o]['name'] for o in organs]
        st.write(organ_list)

        # Selezione organo per DVH
        selected_organ = st.selectbox("Seleziona organo per DVH", organ_list)

        # Trova ID dell'organo selezionato
        organ_id = next(k for k, v in organs.items() if v['name'] == selected_organ)

        # Calcolo DVH
        dvh = dvhcalc.get_dvh(struct, dose, structure_number=organ_id)
        dvh_bins = dvh.bins
        dvh_values = dvh.cumulative / dvh.cumulative.max() * 100  # volume %

        # Visualizza DVH
        fig = px.line(x=dvh_bins, y=dvh_values,
                      labels={'x': 'Dose [Gy]', 'y': 'Volume [%]'},
                      title=f'DVH - {selected_organ}')
        st.plotly_chart(fig)

        # Esempio di plan score semplice
        # (coverage target: organo con nome "PTV", sparo valori fittizi)
        ptv_dvh = dvhcalc.get_dvh(struct, dose, structure_number=organ_id)  # sostituire con PTV
        ptv_coverage = np.mean(ptv_dvh.cumulative / ptv_dvh.cumulative.max() * 100)
        oar_mean = np.mean(dvh_values)  # dose media organo
        score = 0.5 * ptv_coverage + 0.5 * (100 - oar_mean)
        st.write(f"**Plan Score (esempio semplificato): {score:.2f}**")

    except Exception as e:
        st.error(f"Errore nel caricamento dei file DICOM: {e}")
