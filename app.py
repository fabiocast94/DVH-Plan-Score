import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================
# FUNZIONE CONFRONTO VALORI
# ============================================================
def better_value(old, new, criterion):
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    return "Nuovo" if ((criterion == "lower" and new < old) or 
                       (criterion == "higher" and new > old)) else "Vecchio"


# ============================================================
# INTERFACCIA STREAMLIT
# ============================================================
st.title("🔬 Analisi RapidPlan Avanzata – Multi-ID, Multi-Struttura")

uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📁 Anteprima dati")
    st.dataframe(df.head())

    # --------------------------------------------------------
    # SELEZIONE COLONNE
    # --------------------------------------------------------
    st.subheader("⚙️ Seleziona colonne")
    col_id = st.selectbox("Colonna ID paziente", df.columns)
    col_structure = st.selectbox("Colonna struttura (PTV, OAR, ecc.)", df.columns)
    col_metric = st.selectbox("Colonna metrica", df.columns)
    col_old = st.selectbox("Colonna Piano Vecchio", df.columns)
    col_new = st.selectbox("Colonna Piano Nuovo", df.columns)

    # --------------------------------------------------------
    # CRITERI PER METRICHE
    # --------------------------------------------------------
    metrics = df[col_metric].unique()
    structures = df[col_structure].unique()

    st.subheader("📌 Criteri valutazione per ogni metrica")

    criteria = {}
    for m in metrics:
        criteria[m] = st.selectbox(
            f"{m} – quale valore è migliore?",
            ["lower is better", "higher is better"],
            index=0
        )

    # --------------------------------------------------------
    # ANALISI
    # --------------------------------------------------------
    results = []

    for _, row in df.iterrows():
        metric = row[col_metric]
        crit = "lower" if criteria[metric] == "lower is better" else "higher"
        winner = better_value(row[col_old], row[col_new], crit)

        results.append({
            "ID": row[col_id],
            "Struttura": row[col_structure],
            "Metrica": metric,
            "Piano Vecchio": row[col_old],
            "Piano Nuovo": row[col_new],
            "Migliore": winner
        })

    results_df = pd.DataFrame(results)

    st.subheader("📊 Risultati del confronto")
    st.dataframe(results_df)

    # --------------------------------------------------------
    # RADAR PLOT PER ID
    # --------------------------------------------------------
    st.subheader("📈 Radar plot")

    selected_id = st.selectbox("Seleziona ID", df[col_id].unique())
    selected_structure = st.selectbox("Seleziona struttura", structures)

    subset = results_df[
        (results_df["ID"] == selected_id) &
        (results_df["Struttura"] == selected_structure)
    ]

    if subset.empty:
        st.warning("Nessun dato disponibile per questo ID e struttura.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=subset["Piano Vecchio"].values,
            theta=subset["Metrica"].values,
            fill="toself",
            name="Vecchio"
        ))
        fig.add_trace(go.Scatterpolar(
            r=subset["Piano Nuovo"].values,
            theta=subset["Metrica"].values,
            fill="toself",
            name="Nuovo"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )

        st.plotly_chart(fig)

    # --------------------------------------------------------
    # HEATMAP GLOBALE
    # --------------------------------------------------------
    st.subheader("🔥 Heatmap globale (vincitore per ID/Metrica)")

    pivot = results_df.pivot_table(
        index="ID", columns="Metrica", values="Migliore", aggfunc=lambda x: x.iloc[0]
    )

    mapping = {"Vecchio": 0, "Nuovo": 1, "N/A": np.nan}
    heatmap_data = pivot.replace(mapping)

    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=pivot, fmt="", ax=ax)
    st.pyplot(fig2)

    # --------------------------------------------------------
    # ANALISI STATISTICA WILCOXON
    # --------------------------------------------------------
    st.subheader("📊 Test Wilcoxon per ogni metrica")

    wilcoxon_results = []

    for m in metrics:
        old_vals = df[df[col_metric] == m][col_old]
        new_vals = df[df[col_metric] == m][col_new]

        try:
            stat, p_value = wilcoxon(old_vals, new_vals)
            wilcoxon_results.append([m, stat, p_value])
        except:
            wilcoxon_results.append([m, np.nan, np.nan])

    wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=["Metric", "Statistic", "p-value"])
    st.dataframe(wilcoxon_df)

    # --------------------------------------------------------
    # ESPORTA RISULTATI EXCEL
    # --------------------------------------------------------
    st.subheader("📥 Esporta risultati in Excel")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Risultati", index=False)
        pivot.to_excel(writer, sheet_name="Heatmap", index=True)
        wilcoxon_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

    st.download_button(
        label="Scarica file risultati",
        data=output.getvalue(),
        file_name="Confronto_RapidPlan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --------------------------------------------------------
    # STATISTICHE FINALI
    # --------------------------------------------------------
    st.subheader("🏆 Risultato complessivo")

    summary = results_df["Migliore"].value_counts()

    st.write(summary)

    if summary.get("Nuovo", 0) > summary.get("Vecchio", 0):
        st.success("🎉 Il NUOVO modello RapidPlan è globalmente migliore!")
    elif summary.get("Nuovo", 0) < summary.get("Vecchio", 0):
        st.error("⚠️ Il VECCHIO modello RapidPlan risulta globalmente migliore.")
    else:
        st.warning("⚖️ I due modelli risultano equivalenti.")
