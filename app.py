import streamlit as st
import pydicom
import numpy as np
import plotly.express as px

st.title("Plan Quality Dashboard - Multi-Piano DVH")

def get_mask_from_contours_simple(struct, roi_name, dose_shape, dose_spacing, dose_origin):
    """
    Maschera 3D semplificata dell'organo usando bounding box dei contorni.
    """
    roi_number = next(item.ROINumber for item in struct.StructureSetROISequence if item.ROIName==roi_name)
    contours = [c for c in struct.ROIContourSequence if c.ReferencedROINumber==roi_number]
    mask = np.zeros(dose_shape, dtype=bool)

    for contour_seq in contours:
        for cdata in contour_seq.ContourSequence:
            pts = np.array(cdata.ContourData).reshape(-1,3)
            z = int(round((pts[0,2]-dose_origin[2])/dose_spacing[2]))
            if 0 <= z < dose_shape[0]:
                xy_voxels = np.round((pts[:,:2]-dose_origin[:2])/dose_spacing[:2]).astype(int)
                x_min, x_max = xy_voxels[:,0].min(), xy_voxels[:,0].max()
                y_min, y_max = xy_voxels[:,1].min(), xy_voxels[:,1].max()
                mask[z, y_min:y_max+1, x_min:x_max+1] = True
    return mask

st.write("### Carica i piani RTDOSE e il RTSTRUCT")
uploaded_struct = st.file_uploader("RTSTRUCT", type=["dcm"])
uploaded_doses = st.file_uploader("RTDOSE (più file)", type=["dcm"], accept_multiple_files=True)

if uploaded_struct and uploaded_doses:
    try:
        struct = pydicom.dcmread(uploaded_struct)
        roi_names = [item.ROIName for item in struct.StructureSetROISequence]
        selected_organ = st.selectbox("Seleziona organo", roi_names)

        plan_dvhs = {}

        for dose_file in uploaded_doses:
            dose = pydicom.dcmread(dose_file)
            dose_array = dose.pixel_array * dose.DoseGridScaling
            dose_shape = dose_array.shape  # (z,y,x)
            spacing = np.array([float(dose.GridFrameOffsetVector[1]-dose.GridFrameOffsetVector[0])] + list(dose.PixelSpacing))
            origin = np.array([float(dose.ImagePositionPatient[2]), float(dose.ImagePositionPatient[1]), float(dose.ImagePositionPatient[0])])

            mask = get_mask_from_contours_simple(struct, selected_organ, dose_shape, spacing, origin)
            organ_doses = dose_array[mask]

            hist, bin_edges = np.histogram(organ_doses, bins=100, range=(0, np.max(dose_array)))
            dvh_cum = 100 * (1 - np.cumsum(hist)/np.sum(hist))
            plan_dvhs[dose_file.name] = (bin_edges[:-1], dvh_cum)

        # Plot DVH multi-piano
        fig = px.line(title=f"DVH per {selected_organ}")
        for plan_name, (x, y) in plan_dvhs.items():
            fig.add_scatter(x=x, y=y, mode='lines', name=plan_name)
        fig.update_layout(xaxis_title="Dose [Gy]", yaxis_title="Volume [%]")
        st.plotly_chart(fig)

        # Calcolo plan score semplice (dose media)
        st.write("### Plan Scores (dose media dell'organo)")
        for plan_name, (x, y) in plan_dvhs.items():
            mean_dose = np.mean(dose_array[mask])
            plan_score = max(0, 100 - mean_dose)  # esempio semplice
            st.write(f"{plan_name}: {plan_score:.2f}")

    except Exception as e:
        st.error(f"Errore: {e}")
