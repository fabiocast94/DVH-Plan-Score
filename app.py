import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================
# 1) CRITERI FISSI PER OGNI METRICA
#    (modificali tu quando vuoi → restano permanenti)
# ============================================================
METRIC_CRITERIA = {
    "D95": "higher",
    "D98": "higher",
    "D2": "lower",
    "Dmax": "lower",
    "Dmean": "lower",
    "V95": "higher",
    "V107": "lower",
    # aggiungerai metriche reali dopo aver visto il file completo
}


# ============================================================
# FUNZIONE: confronta due valori secondo criterio
# ============================================================
def better_value(old, new, metric):
    crit = METRIC_CRITERIA.get(metric, "lower")   # default: lower is better
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"


# ============================================================
# STREAMLIT APP
# ============================================================
st.title("🔬 Analisi Dose Hunter – Confronto Pianificazione RapidPlan")
st.write("Versione con criteri fissi e riconoscimento automatico del piano 'new'.")

uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

if uploaded_file:
    # ============================================================
    # 2) IMPORTAZIONE FILE DOSE HUNTER
    # ============================================================
    df = pd.read_excel(uploaded_file)

    st.subheader("📁 Anteprima dati")
    st.dataframe(df.head())

    # ============================================================
    # 3) IDENTIFICAZIONE AUTOMATICA DELLE COLONNE
    # ============================================================
    col_id = "ID"
    col_plan = "planID"
    col_volume = "Volume"

    # La metrica è la colonna successiva al Volume
    idx_volume = df.columns.get_loc(col_volume)
    col_metric = df.columns[idx_volume + 1]

    st.info(f"Metrica identificata automaticamente: **{col_metric}**")

    # ============================================================
    # 4) IDENTIFICAZIONE AUTOMATICA DEI PIANI
    # ============================================================
    df["Type"] = df[col_plan].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

    # Raggruppiamo per ID + metrica
    grouped = df.groupby([col_id, col_metric])

    results = []

    # ============================================================
    # 5) GESTIONE MULTIPARAMETRICA AUTOMATICA
    # ============================================================
    for (id_val, metric), group in grouped:
        if len(group) != 2:
            continue   # serve 1 vecchio + 1 nuovo

        val_old = group[group["Type"] == "Vecchio"].iloc[0][col_volume]
        val_new = group[group["Type"] == "Nuovo"].iloc[0][col_volume]

        winner = better_value(val_old, val_new, metric)

        results.append({
            "ID": id_val,
            "Metrica": metric,
            "Piano Vecchio": val_old,
            "Piano Nuovo": val_new,
            "Migliore": winner
        })

    results_df = pd.DataFrame(results)

    st.subheader("📊 Risultati del confronto")
    st.dataframe(results_df)

    # ============================================================
    # 6) RADAR PLOT AUTOMATICO
    # ============================================================
    st.subheader("📈 Radar plot (per ID)")

    selected_id = st.selectbox("Seleziona ID", results_df["ID"].unique())

    sub = results_df[results_df["ID"] == selected_id]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sub["Piano Vecchio"].values,
        theta=sub["Metrica"].values,
        fill='toself',
        name="Vecchio"
    ))
    fig.add_trace(go.Scatterpolar(
        r=sub["Piano Nuovo"].values,
        theta=sub["Metrica"].values,
        fill='toself',
        name="Nuovo"
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)

    # ============================================================
    # 7) HEATMAP GLOBALE
    # ============================================================
    st.subheader("🔥 Heatmap globale – piano migliore per ID/Parametro")

    pivot = results_df.pivot(index="ID", columns="Metrica", values="Migliore")

    mapping = {"Vecchio": 0, "Nuovo": 1}
    heatmap_data = pivot.replace(mapping)

    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=pivot, fmt="", ax=ax)
    st.pyplot(fig2)

    # ============================================================
    # 8) TEST STATISTICO WILCOXON (per metrica)
    # ============================================================
    st.subheader("📊 Test Wilcoxon")

    wilcox = []

    for metric in results_df["Metrica"].unique():
        d = results_df[results_df["Metrica"] == metric]
        try:
            stat, pval = wilcoxon(d["Piano Vecchio"], d["Piano Nuovo"])
            wilcox.append([metric, stat, pval])
        except:
            wilcox.append([metric, None, None])

    wilcox_df = pd.DataFrame(wilcox, columns=["Metrica", "Statistic", "p-value"])
    st.dataframe(wilcox_df)

    # ============================================================
    # 9) ESPORTAZIONE RISULTATI
    # ============================================================
    st.subheader("📥 Esporta risultati")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Risultati", index=False)
        pivot.to_excel(writer, sheet_name="Heatmap")
        wilcox_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

    st.download_button(
        label="Scarica Excel",
        data=output.getvalue(),
        file_name="Analisi_RapidPlan.xlsx"
    )

    # ============================================================
    # 10) RISULTATO FINALE
    # ============================================================
    st.subheader("🏆 Risultato finale")

    count = results_df["Migliore"].value_counts()

    if count.get("Nuovo", 0) > count.get("Vecchio", 0):
        st.success("🎉 Il NUOVO modello è globalmente migliore!")
    else:
        st.error("⚠️ Il modello vecchio sembra migliore… per ora.")
