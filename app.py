import streamlit as st
import pydicom
import numpy as np
import plotly.express as px

st.title("Plan Quality Dashboard - RTDOSE/RTSTRUCT")

uploaded_struct = st.file_uploader("Carica RTSTRUCT", type=["dcm"])
uploaded_dose = st.file_uploader("Carica RTDOSE", type=["dcm"])

if uploaded_struct and uploaded_dose:
    try:
        # Carica RTDOSE
        dose = pydicom.dcmread(uploaded_dose)
        dose_array = dose.pixel_array * dose.DoseGridScaling  # dose in Gy

        # Carica RTSTRUCT
        struct = pydicom.dcmread(uploaded_struct)

        # Lista delle ROI (nomi degli organi)
        roi_names = [item.ROIName for item in struct.StructureSetROISequence]
        st.subheader("Organi disponibili nel piano")
        st.write(roi_names)

        # Seleziona organo
        selected_organ = st.selectbox("Seleziona organo per DVH", roi_names)

        # Trova ROI ID
        roi_number = next(item.ROINumber for item in struct.StructureSetROISequence if item.ROIName == selected_organ)

        # Nota: per semplicità, qui non calcoliamo DVH reale senza contorni 3D
        # Creiamo DVH fittizio come esempio
        dvh_bins = np.linspace(0, np.max(dose_array), 100)
        dvh_values = 100 * np.exp(-0.05 * dvh_bins)  # volume % decrescente

        # Grafico DVH
        fig = px.line(x=dvh_bins, y=dvh_values,
                      labels={'x': 'Dose [Gy]', 'y': 'Volume [%]'},
                      title=f'DVH - {selected_organ}')
        st.plotly_chart(fig)

        # Plan score fittizio
        plan_score = np.random.uniform(50, 100)
        st.write(f"**Plan Score (esempio semplificato): {plan_score:.2f}**")

    except Exception as e:
        st.error(f"Errore nel caricamento dei file DICOM: {e}")
