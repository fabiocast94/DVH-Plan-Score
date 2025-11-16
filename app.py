import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================
# 1) CRITERI FISSI PER LE METRICHE
#    (modificali una sola volta, poi restano permanenti)
# ============================================================
METRIC_CRITERIA = {
    "D95": "higher",
    "D98": "higher",
    "D2": "lower",
    "D50": "lower",
    "Dmax": "lower",
    "Dmean": "lower",
    "V95": "higher",
    "V90": "higher",
    "V107": "lower",
    "V20": "lower",
    "V5": "lower",
    "V10": "lower",
    # altre metriche le aggiungeremo quando vediamo il file finale
}


# ============================================================
# Compara due valori secondo il criterio della metrica
# ============================================================
def better_value(old, new, metric):
    crit = METRIC_CRITERIA.get(metric, "lower")  # default = "minore è meglio"
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"


# ============================================================
# STREAMLIT
# ============================================================
st.title("🔬 Analisi Dose Hunter – Multi-Struttura e Multi-Metrica")
st.write("Versione con criteri fissi e riconoscimento automatico del piano 'new'.")

uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

if uploaded_file:

    # ============================================================
    # 2) IMPORTA FILE
    # ============================================================
    df = pd.read_excel(uploaded_file)

    st.subheader("📁 Anteprima dati")
    st.dataframe(df.head())

    # ============================================================
    # 3) IDENTIFICAZIONE AUTOMATICA DELLE COLONNE
    # ============================================================
    col_id = "ID"
    col_plan = "planID"
    col_structure = "Struttura"
    col_volume = "Volume"

    # Metrica = colonna successiva a Volume
    idx_volume = df.columns.get_loc(col_volume)
    col_metric = df.columns[idx_volume + 1]
    col_value = col_metric  # stesso per Dose Hunter

    st.info(f"Metrica identificata automaticamente: **{col_metric}**")

    # ============================================================
    # 4) IDENTIFICAZIONE VECCHIO/Nuovo
    # ============================================================
    df["TipoPiano"] = df[col_plan].apply(
        lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio"
    )

    # ============================================================
    # 5) RACCOLTA RISULTATI MULTI-STRUTTURA
    # ============================================================
    grouped = df.groupby([col_id, col_structure, col_metric])

    results = []

    for (id_val, struct, metric), group in grouped:

        if len(group) != 2:
            continue  # serve 1 vecchio + 1 nuovo

        old_value = group[group["TipoPiano"] == "Vecchio"].iloc[0][col_value]
        new_value = group[group["TipoPiano"] == "Nuovo"].iloc[0][col_value]

        winner = better_value(old_value, new_value, metric)

        results.append({
            "ID": id_val,
            "Struttura": struct,
            "Metrica": metric,
            "Valore Vecchio": old_value,
            "Valore Nuovo": new_value,
            "Migliore": winner
        })

    results_df = pd.DataFrame(results)

    st.subheader("📊 Risultati del confronto")
    st.dataframe(results_df)

    # ============================================================
    # 6) RADAR PLOT PER ID + STRUTTURA
    # ============================================================
    st.subheader("📈 Radar Plot (per ID e Struttura)")

    selected_id = st.selectbox("Seleziona ID:", results_df["ID"].unique())
    subset_structures = results_df[results_df["ID"] == selected_id]["Struttura"].unique()
    selected_structure = st.selectbox("Seleziona struttura:", subset_structures)

    radar_data = results_df[
        (results_df["ID"] == selected_id) &
        (results_df["Struttura"] == selected_structure)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_data["Valore Vecchio"],
        theta=radar_data["Metrica"],
        fill='toself',
        name="Vecchio"
    ))
    fig.add_trace(go.Scatterpolar(
        r=radar_data["Valore Nuovo"],
        theta=radar_data["Metrica"],
        fill='toself',
        name="Nuovo"
    ))

    st.plotly_chart(fig)

    # ============================================================
    # 7) HEATMAP GLOBALE (ID × Metrica × Struttura)
    # ============================================================
    st.subheader("🔥 Heatmap globale")

    pivot = results_df.pivot_table(
        index="ID",
        columns=["Struttura", "Metrica"],
        values="Migliore",
        aggfunc=lambda x: x.iloc[0]
    )

    heatmap_numeric = pivot.replace({"Vecchio": 0, "Nuovo": 1})

    fig2, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(heatmap_numeric, cmap="coolwarm", annot=pivot, fmt="", ax=ax)
    st.pyplot(fig2)

    # ============================================================
    # 8) WILCOXON PER STRUTTURA + METRICA
    # ============================================================
    st.subheader("📊 Test Wilcoxon (per Struttura e Metrica)")

    wilcox_out = []

    for struct in results_df["Struttura"].unique():
        for metric in results_df["Metrica"].unique():

            values = results_df[
                (results_df["Struttura"] == struct) &
                (results_df["Metrica"] == metric)
            ]

            if len(values) < 2:
                continue

            try:
                stat, pval = wilcoxon(values["Valore Vecchio"], values["Valore Nuovo"])
                wilcox_out.append([struct, metric, stat, pval])
            except:
                wilcox_out.append([struct, metric, None, None])

    wilcox_df = pd.DataFrame(wilcox_out, columns=["Struttura", "Metrica", "Statistic", "p-value"])
    st.dataframe(wilcox_df)

    # ============================================================
    # 9) EXPORT COMPLETO
    # ============================================================
    st.subheader("📥 Esporta risultati")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Risultati", index=False)
        pivot.to_excel(writer, sheet_name="Heatmap")
        wilcox_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

    st.download_button(
        label="Scarica Excel completo",
        data=output.getvalue(),
        file_name="Analisi_RapidPlan_multiStruttura.xlsx"
    )

    # ============================================================
    # 10) RISULTATO FINALE
    # ============================================================
    st.subheader("🏆 RISULTATO FINALE")

    summary = results_df["Migliore"].value_counts()

    if summary.get("Nuovo", 0) > summary.get("Vecchio", 0):
        st.success("🎉 Il nuovo modello RapidPlan è globalmente migliore!")
    else:
        st.error("⚠️ Il vecchio modello sembra migliore… almeno con questi dati.")
