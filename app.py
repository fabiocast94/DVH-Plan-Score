import streamlit as st
import pydicom
from rt_utils import RTStructBuilder
import numpy as np
import plotly.express as px

st.title("Plan Quality Dashboard - RTDOSE/RTSTRUCT")

# Caricamento dei file DICOM
uploaded_struct = st.file_uploader("Carica RTSTRUCT", type=["dcm"])
uploaded_dose = st.file_uploader("Carica RTDOSE", type=["dcm"])

if uploaded_struct and uploaded_dose:
    try:
        # Lettura RTDOSE
        dose = pydicom.dcmread(uploaded_dose)
        dose_array = dose.pixel_array * dose.DoseGridScaling  # dose in Gy

        # Lettura RTSTRUCT
        struct = RTStructBuilder.create_from(dicom_series=[], rt_struct_file=uploaded_struct)

        # Lista degli organi
        organ_names = struct.get_roi_names()
        st.subheader("Organi disponibili nel piano")
        st.write(organ_names)

        # Selezione organo
        selected_organ = st.selectbox("Seleziona organo per DVH", organ_names)

        # Maschera 3D dell'organo
        organ_mask = struct.get_roi_mask_by_name(selected_organ)

        # Calcolo DVH (cumulative)
        organ_doses = dose_array[organ_mask > 0].flatten()
        hist, bin_edges = np.histogram(organ_doses, bins=100, range=(0, np.max(organ_doses)))
        dvh_cumulative = 100 * (1 - np.cumsum(hist)/np.sum(hist))  # volume %

        # Grafico DVH
        fig = px.line(x=bin_edges[:-1], y=dvh_cumulative,
                      labels={'x':'Dose [Gy]', 'y':'Volume [%]'},
                      title=f'DVH - {selected_organ}')
        st.plotly_chart(fig)

        # Esempio plan score semplificato
        # Assumiamo che PTV sia presente
        if "PTV" in organ_names:
            ptv_mask = struct.get_roi_mask_by_name("PTV")
            ptv_doses = dose_array[ptv_mask > 0].flatten()
            ptv_coverage = np.mean(ptv_doses >= 95) * 100  # % volume >= 95 Gy
        else:
            ptv_coverage = 0

        oar_mean_dose = np.mean(organ_doses)  # dose media organo selezionato
        plan_score = 0.5 * ptv_coverage + 0.5 * (100 - oar_mean_dose)
        st.write(f"**Plan Score (esempio semplificato): {plan_score:.2f}**")

    except Exception as e:
        st.error(f"Errore nel caricamento dei file DICOM: {e}")
