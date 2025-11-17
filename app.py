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

EQUIV_THRESHOLD = 0.01

def better_value(old, new, metric):
    if pd.isna(old) or pd.isna(new): return "N/A"
    crit = METRIC_CRITERIA.get(metric,"lower")
    rel_diff = abs(new - old) / old if old != 0 else 0
    if rel_diff < EQUIV_THRESHOLD:
        return "Equivalente"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"

# ============================================================
st.title("🔬 Analisi Dose Hunter – Multi-Struttura e Multi-Metrica")
uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

# ============================================================
# Preset per distretti
PRESET_STRUCTURES = {
    "Thorax": ["PTV", "Heart", "Lung"],
    "Head and Neck": ["PTV", "SpinalCord", "Parotid_L", "Parotid_R"],
    "Breast": ["PTV", "Heart", "Lung"],
    "Abdomen": ["PTV", "Liver", "Kidney_L", "Kidney_R"],
    "Prostate": ["PTV", "Bladder", "Rectum"],
    "Pelvi": ["PTV", "Bladder", "Rectum", "Femur_L", "Femur_R"]
}

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ============================================================
    # Identifica colonne vol
    vol_cols = [c for c in df.columns if "(vol)" in c.lower()]
    df["Struttura"] = np.nan

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
            name = col.split("(")[0].strip()  # semplice estrazione nome metrica
            metrics.append(name)
            m_to_c[name] = col
        metric_map[struct] = metrics
        metric_column_map[struct] = m_to_c

    # ============================================================
    # Tipo piano Nuovo vs Vecchio
    plan_col = [c for c in df.columns if "plan" in c.lower()][0]
    df["TipoPiano"] = df[plan_col].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

    # ============================================================
    # Costruisci results_df
    results = []
    id_col = [c for c in df.columns if "id" in c.lower()][0]
    for id_val in df[id_col].unique():
        temp = df[df[id_col]==id_val]
        for struct, metrics in metric_map.items():
            sub = temp[temp["Struttura"]==struct]
            if sub.empty: continue
            for m in metrics:
                col = metric_column_map[struct][m]
                v_old = sub[sub["TipoPiano"]=="Vecchio"][col].iloc[0]
                v_new = sub[sub["TipoPiano"]=="Nuovo"][col].iloc[0]
                winner = better_value(v_old,v_new,m)
                diff_pct = ((v_new - v_old)/v_old*100 if v_old!=0 else 0)
                results.append({
                    "ID": id_val,
                    "Struttura": struct,
                    "Metrica": m,
                    "Valore Vecchio": v_old,
                    "Valore Nuovo": v_new,
                    "Δ %": diff_pct,
                    "Migliore": winner
                })

    if not results:
        st.warning("❌ Nessuna struttura trovata! Controlla il file Excel.")
        st.stop()

    results_df = pd.DataFrame(results)

    # ============================================================
    # Filtri interattivi
    st.sidebar.header("🔍 Filtri")
    distretto_sel = st.sidebar.selectbox("Seleziona preset distretto", list(PRESET_STRUCTURES.keys()))
    default_structs = PRESET_STRUCTURES[distretto_sel]

    structs_sel = st.sidebar.multiselect(
        "Seleziona strutture",
        sorted(results_df["Struttura"].unique()),
        default=default_structs
    )

    metrics_sel = st.sidebar.multiselect(
        "Seleziona metriche",
        sorted(results_df["Metrica"].unique()),
        default=None
    )

    results_filtered = results_df.copy()
    if structs_sel:
        results_filtered = results_filtered[results_filtered["Struttura"].isin(structs_sel)]
    if metrics_sel:
        results_filtered = results_filtered[results_filtered["Metrica"].isin(metrics_sel)]

    # ============================================================
    # Separazione PTV vs OAR
    PTV_df = results_filtered[results_filtered["Struttura"].str.contains("PTV", case=False)]
    OAR_df = results_filtered[~results_filtered["Struttura"].str.contains("PTV", case=False)]

    st.subheader("📊 Risultati PTV")
    st.dataframe(PTV_df)
    st.subheader("🫁 Risultati OAR")
    st.dataframe(OAR_df)

    # ============================================================
    # Wilcoxon
    wilcox = []
    for struct in results_filtered["Struttura"].unique():
        for met in results_filtered["Metrica"].unique():
            vals = results_filtered[(results_filtered["Struttura"]==struct)&(results_filtered["Metrica"]==met)]
            if len(vals) < 2: continue
            try:
                stat,p = wilcoxon(vals["Valore Vecchio"], vals["Valore Nuovo"])
            except:
                stat,p = None,None
            wilcox.append([struct,met,stat,p])

    wilcox_df = pd.DataFrame(wilcox,columns=["Struttura","Metrica","Statistic","p-value"])
    wilcox_df["Significativo"] = wilcox_df["p-value"] < 0.05

    show_only_sig = st.sidebar.checkbox("Mostra solo metriche significative (p < 0.05)")
    if show_only_sig:
        results_filtered = results_filtered.merge(
            wilcox_df[wilcox_df["Significativo"]],on=["Struttura","Metrica"]
        )

    st.subheader("📊 Risultati Wilcoxon")
    st.dataframe(wilcox_df)

    # ============================================================
    # Export Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        PTV_df.to_excel(writer,"PTV",index=False)
        OAR_df.to_excel(writer,"OAR",index=False)
        wilcox_df.to_excel(writer,"Wilcoxon",index=False)
    st.download_button(
        "📥 Scarica Excel completo",
        data = output.getvalue(),
        file_name="Analisi_RapidPlan_Advanced.xlsx"
    )

    # ============================================================
    # Risultato finale
    st.subheader("🏁 RISULTATO FINALE")
    summary = results_filtered["Migliore"].value_counts()
    st.write(summary)
    if summary.get("Nuovo",0) > summary.get("Vecchio",0):
        st.success("🎉 Il nuovo modello RapidPlan risulta complessivamente migliore!")
    else:
        st.error("⚠️ Il modello vecchio sembra migliore su queste metriche.")
