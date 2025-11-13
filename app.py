import streamlit as st
import pydicom
import numpy as np
import plotly.express as px
from scipy.spatial import Delaunay

st.title("Plan Quality Dashboard - Real DVH")

def get_mask_from_contours(struct, roi_name, dose_shape, dose_spacing, dose_origin):
    """
    Crea una maschera 3D boolean di un organo dal RTSTRUCT.
    struct: RTSTRUCT pydicom object
    roi_name: nome ROI
    dose_shape: shape array dose (z,y,x)
    dose_spacing: pixel spacing (x,y,z)
    dose_origin: origin (x0,y0,z0)
    """
    import itertools

    # Trova ROI number
    roi_number = next(item.ROINumber for item in struct.StructureSetROISequence if item.ROIName==roi_name)

    # Trova contours associati
    contours = [c for c in struct.ROIContourSequence if c.ReferencedROINumber==roi_number]
    mask = np.zeros(dose_shape, dtype=bool)

    for contour_seq in contours:
        for cdata in contour_seq.ContourSequence:
            pts = np.array(cdata.ContourData).reshape(-1,3)
            z = int(round((pts[0,2]-dose_origin[2])/dose_spacing[2]))
            if 0<=z<dose_shape[0]:
                # Convert x,y in voxel indices
                xy_voxels = np.round((pts[:,:2]-dose_origin[:2])/dose_spacing[:2]).astype(int)
                # Simple polygon fill using Delaunay triangulation
                try:
                    tri = Delaunay(xy_voxels)
                    for i,j in itertools.product(range(dose_shape[1]), range(dose_shape[2])):
                        if tri.find_simplex(np.array([[i,j]]))>=0:
                            mask[z,i,j]=True
                except:
                    pass
    return mask

# Caricamento file
uploaded_struct = st.file_uploader("Carica RTSTRUCT", type=["dcm"])
uploaded_dose = st.file_uploader("Carica RTDOSE", type=["dcm"])

if uploaded_struct and uploaded_dose:
    try:
        dose = pydicom.dcmread(uploaded_dose)
        struct = pydicom.dcmread(uploaded_struct)

        dose_array = dose.pixel_array * dose.DoseGridScaling
        dose_shape = dose_array.shape  # (z,y,x)
        spacing = np.array([float(dose.GridFrameOffsetVector[1]-dose.GridFrameOffsetVector[0])] + list(dose.PixelSpacing) )
        origin = np.array([float(dose.ImagePositionPatient[2]), float(dose.ImagePositionPatient[1]), float(dose.ImagePositionPatient[0])])

        roi_names = [item.ROIName for item in struct.StructureSetROISequence]
        selected_organ = st.selectbox("Seleziona organo", roi_names)

        mask = get_mask_from_contours(struct, selected_organ, dose_shape, spacing, origin)

        # Calcolo DVH
        organ_doses = dose_array[mask]
        hist, bin_edges = np.histogram(organ_doses, bins=100, range=(0, np.max(dose_array)))
        dvh_cum = 100 * (1 - np.cumsum(hist)/np.sum(hist))

        # Plot DVH
        fig = px.line(x=bin_edges[:-1], y=dvh_cum, labels={'x':'Dose [Gy]','y':'Volume [%]'}, title=f'DVH {selected_organ}')
        st.plotly_chart(fig)

        # Plan score semplificato
        mean_dose = np.mean(organ_doses)
        plan_score = max(0, 100 - mean_dose)  # esempio
        st.write(f"**Plan Score: {plan_score:.2f}**")

    except Exception as e:
        st.error(f"Errore: {e}")
