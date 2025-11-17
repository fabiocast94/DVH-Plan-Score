import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from io import BytesIO

# ============================================================
# 1) CRITERI METRICHE
# ============================================================
METRIC_CRITERIA = {
    "HI": "lower", "D95": "higher", "D98": "higher", "D2": "lower",
    "D50": "lower", "Dmax": "lower", "Dmean": "lower",
    "V95": "higher","V90": "higher","V107": "lower",
    "V20": "lower","V5": "lower","V10": "lower",
    "CI": "higher"
}

EQUIV_THRESHOLD = 0.01  ### 1% threshold

def better_value(old, new, metric):
    if pd.isna(old) or pd.isna(new): return "N/A"
    crit = METRIC_CRITERIA.get(metric,"lower")
    rel_diff = abs(new - old) / old if old != 0 else 0
    if rel_diff < EQUIV_THRESHOLD:
        return "Equivalent"
    if crit == "lower":
        return "New" if new < old else "Old"
    else:
        return "New" if new > old else "Old"

# ============================================================
# Preset structures (English names)
# ============================================================
PRESET_STRUCTURES = {
    "PTV": ["PTV_High", "PTV_Low", "PTV_Boost"],
    "OAR": ["Lung_L", "Lung_R", "Heart", "SpinalCord", "Esophagus", "Breast"]
}

ALL_STRUCTURES = [s for lst in PRESET_STRUCTURES.values() for s in lst]

# ============================================================
st.title("🔬 Dose Hunter Analysis – Multi-Structure & Multi-Metric")

uploaded_file = st.file_uploader("Upload Excel file Dose Hunter", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ============================================================
    # IDENTIFY COLUMNS
    # ============================================================
    possible_id = [c for c in df.columns if "id" in c.lower()]
    col_id = possible_id[0] if possible_id else None

    possible_plan = [c for c in df.columns if "plan" in c.lower()]
    col_plan = possible_plan[0] if possible_plan else None

    vol_cols = [c for c in df.columns if "(vol)" in c.lower()]
    df["Struttura"] = None

    metric_map = {}
    metric_column_map = {}

    for i, vol in enumerate(vol_cols):
        struct = vol.replace("(vol)", "").strip()
        mask = df[vol].notna()
        df.loc[mask, "Struttura"] = struct

        start = df.columns.get_loc(vol)+1
        end = df.columns.get_loc(vol_cols[i+1]) if i+1 < len(vol_cols) else df.shape[1]

        metrics = []
        m_to_c = {}

        for col in df.columns[start:end]:
            if not pd.api.types.is_numeric_dtype(df[col]): continue
            name = col.split("(")[1].split(")")[0] if "(" in col else col
            metrics.append(name)
            m_to_c[name] = col

        metric_map[struct] = metrics
        metric_column_map[struct] = m_to_c

    # ============================================================
    # NEW vs OLD
    # ============================================================
    if col_plan:
        df["PlanType"] = df[col_plan].apply(lambda x: "New" if "new" in str(x).lower() else "Old")
    else:
        df["PlanType"] = "Unknown"

    results = []
    if col_id:
        for id_val in df[col_id].unique():
            temp = df[df[col_id]==id_val]

            for struct, metrics in metric_map.items():
                sub = temp[temp["Struttura"]==struct]
                if sub.empty: continue

                for m in metrics:
                    col = metric_column_map[struct][m]

                    v_old = sub[sub["PlanType"]=="Old"][col].iloc[0] if not sub[sub["PlanType"]=="Old"].empty else np.nan
                    v_new = sub[sub["PlanType"]=="New"][col].iloc[0] if not sub[sub["PlanType"]=="New"].empty else np.nan

                    winner = better_value(v_old,v_new,m)
                    diff_pct = ((v_new - v_old)/v_old*100 if v_old!=0 else 0)

                    results.append({
                        "ID": id_val,
                        "Structure": struct,
                        "Metric": m,
                        "Old Value": v_old,
                        "New Value": v_new,
                        "Δ %": diff_pct,
                        "Better": winner
                    })

    results_df = pd.DataFrame(results)

    # ============================================================
    # FILTERS (always show all preset structures)
    # ============================================================
    st.sidebar.header("🔍 Filters")

    structs_sel = st.sidebar.multiselect(
        "Select structures",
        options=ALL_STRUCTURES,
        default=None
    )

    metrics_sel = st.sidebar.multiselect(
        "Select metrics",
        options=results_df["Metric"].unique(),
        default=None
    )

    results_filtered = results_df.copy()
    if structs_sel:
        results_filtered = results_filtered[results_filtered["Structure"].isin(structs_sel)]
    if metrics_sel:
        results_filtered = results_filtered[results_filtered["Metric"].isin(metrics_sel)]

    # ============================================================
    # PTV vs OAR separation
    # ============================================================
    PTV_df = results_filtered[results_filtered["Structure"].str.contains("PTV", case=False)]
    OAR_df = results_filtered[~results_filtered["Structure"].str.contains("PTV", case=False)]

    st.subheader("📊 PTV Results")
    st.dataframe(PTV_df)

    st.subheader("🫁 OAR Results")
    st.dataframe(OAR_df)

    # ============================================================
    # Wilcoxon test
    # ============================================================
    wilcox = []
    for struct in results_filtered["Structure"].unique():
        for met in results_filtered["Metric"].unique():
            vals = results_filtered[(results_filtered["Structure"]==struct)&(results_filtered["Metric"]==met)]
            if len(vals) < 2: continue
            try:
                stat,p = wilcoxon(vals["Old Value"], vals["New Value"])
            except:
                stat,p = None,None
            wilcox.append([struct,met,stat,p])

    wilcox_df = pd.DataFrame(wilcox,columns=["Structure","Metric","Statistic","p-value"])
    wilcox_df["Significant"] = wilcox_df["p-value"] < 0.05

    show_only_sig = st.sidebar.checkbox("Show only significant metrics (p < 0.05)")
    if show_only_sig:
        results_filtered = results_filtered.merge(
            wilcox_df[wilcox_df["Significant"]],on=["Structure","Metric"]
        )

    st.subheader("📊 Wilcoxon Results")
    st.dataframe(wilcox_df)

    # ============================================================
    # Export Excel
    # ============================================================
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        PTV_df.to_excel(writer,"PTV",index=False)
        OAR_df.to_excel(writer,"OAR",index=False)
        wilcox_df.to_excel(writer,"Wilcoxon",index=False)

    st.download_button(
        "📥 Download full Excel",
        data = output.getvalue(),
        file_name="DoseHunter_Analysis.xlsx"
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    st.subheader("🏁 FINAL RESULT")

    summary = results_filtered["Better"].value_counts()
    st.write(summary)

    if summary.get("New",0) > summary.get("Old",0):
        st.success("🎉 The new RapidPlan model is overall better!")
    else:
        st.error("⚠️ The old model seems better based on these metrics.")
