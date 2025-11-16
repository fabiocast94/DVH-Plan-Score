import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# 1) CRITERI FISSI PER LE METRICHE (MODIFICABILI QUI)
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
}

def better_value(old, new, metric):
    crit = METRIC_CRITERIA.get(metric, "lower")
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"

# ============================================================
# 2) STREAMLIT UI
# ============================================================
st.title("🔬 Analisi Dose Hunter – Multi-Struttura e Multi-Metrica")
st.write("Identificazione automatica della struttura dalla colonna '(vol)'.")

uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📁 Anteprima dati")
    st.dataframe(df.head())

    st.write("🔎 **Colonne trovate nel file:**")
    st.write(list(df.columns))

    # ============================================================
    # 3) IDENTIFICAZIONE AUTOMATICA COLONNE
    # ============================================================

    # ID
    possible_id_cols = [c for c in df.columns if "id" in c.lower()]
    col_id = possible_id_cols[0] if possible_id_cols else df.columns[0]
    st.info(f"Colonna ID identificata: **{col_id}**")

    # Piano
    possible_plan_cols = [c for c in df.columns if "plan" in c.lower()]
    if possible_plan_cols:
        col_plan = possible_plan_cols[0]
    else:
        st.error("❌ Nessuna colonna piano trovata.")
        st.stop()
    st.info(f"Colonna Piano identificata: **{col_plan}**")

    # Volume / Struttura
    possible_vol_cols = [c for c in df.columns if "vol" in c.lower()]
    if possible_vol_cols:
        col_volume = possible_vol_cols[0]
        st.info(f"Colonna Volume identificata: **{col_volume}**")

        # Estrazione nome struttura dal nome colonna
        struttura_nome = col_volume.replace("(vol)", "").strip()
        df["Struttura"] = struttura_nome

        # Metrica = colonna successiva
        idx_volume = df.columns.get_loc(col_volume)
        if idx_volume + 1 < len(df.columns):
            col_metric = df.columns[idx_volume + 1]
        else:
            st.error("❌ Nessuna colonna metrica trovata dopo Volume.")
            st.stop()
    else:
        st.error("❌ Nessuna colonna con 'vol' trovata.")
        st.stop()

    col_value = col_metric
    st.success(f"Metrica identificata automaticamente: **{col_metric}**")

    # ============================================================
    # 4) IDENTIFICAZIONE VECCHIO VS NUOVO
    # ============================================================
    df["TipoPiano"] = df[col_plan].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

    # ============================================================
    # 5) COSTRUZIONE RISULTATI MULTI-STRUTTURA (robusta)
    # ============================================================
    results = []

    for id_val in df[col_id].unique():
        for struct in df["Struttura"].unique():
            subset = df[(df[col_id] == id_val) & (df["Struttura"] == struct)]
            if len(subset) < 2:
                continue
            old_row = subset[subset["TipoPiano"] == "Vecchio"]
            new_row = subset[subset["TipoPiano"] == "Nuovo"]
            if old_row.empty or new_row.empty:
                continue
            old_value = old_row.iloc[0][col_value]
            new_value = new_row.iloc[0][col_value]
            winner = better_value(old_value, new_value, col_metric)
            results.append({
                "ID": id_val,
                "Struttura": struct,
                "Metrica": col_metric,
                "Valore Vecchio": old_value,
                "Valore Nuovo": new_value,
                "Migliore": winner
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        st.warning("⚠️ Nessun dato valido trovato per il confronto Vecchio/Nuovo.")
        st.stop()

    st.subheader("📊 Risultati del confronto")
    st.dataframe(results_df)

    # ============================================================
    # 6) RADAR PLOT
    # ============================================================
    st.subheader("📈 Radar Plot per ID e Struttura")
    selected_id = st.selectbox("Seleziona ID", results_df["ID"].unique())
    subset_structures = results_df[results_df["ID"] == selected_id]["Struttura"].unique()
    selected_structure = st.selectbox("Seleziona Struttura", subset_structures)

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
    # 7) HEATMAP
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
    # 8) WILCOXON
    # ============================================================
    st.subheader("📊 Test Wilcoxon")
    wilcox_out = []
    for struct in results_df["Struttura"].unique():
        for metric in results_df["Metrica"].unique():
            values = results_df[(results_df["Struttura"] == struct) &
                                (results_df["Metrica"] == metric)]
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
    # 9) EXPORT EXCEL
    # ============================================================
    st.subheader("📥 Esporta risultati")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Risultati", index=False)
        pivot.to_excel(writer, sheet_name="Heatmap")
        wilcox_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

    st.download_button(
        "Scarica Excel completo",
        data=output.getvalue(),
        file_name="Analisi_RapidPlan_multiStruttura.xlsx"
    )

    # ============================================================
    # 10) RISULTATO FINALE
    # ============================================================
    st.subheader("🏆 RISULTATO FINALE")
    summary = results_df["Migliore"].value_counts()
    st.write(summary)
    if summary.get("Nuovo", 0) > summary.get("Vecchio", 0):
        st.success("🎉 Il nuovo modello RapidPlan risulta migliore!")
    else:
        st.error("⚠️ Il vecchio modello sembra migliore… per questi dati.")
