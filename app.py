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
    "HI": "lower",        # Homogeneity Index
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
    "CI": "higher",       # Conformity Index
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
st.write("Identificazione automatica delle strutture e delle metriche dal file Dose Hunter.")

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

    # Trova tutte le colonne volume (strutture)
    vol_cols = [c for c in df.columns if "(vol)" in c.lower()]
    if not vol_cols:
        st.error("❌ Nessuna colonna volume trovata.")
        st.stop()

    # ============================================================
    # 4) COSTRUZIONE STRUTTURE E METRICHE
    # ============================================================
    df["Struttura"] = ""
    metric_map = {}  # struttura -> lista metriche
    metric_column_map = {}  # struttura -> dict metric -> colonna originale

    for i, vol_col in enumerate(vol_cols):
        struttura_nome = vol_col.replace("(vol)", "").strip()
        df.loc[:, "Struttura"] = struttura_nome if df["Struttura"].eq("").all() else df["Struttura"]

        idx_start = df.columns.get_loc(vol_col) + 1
        idx_end = df.shape[1]
        if i + 1 < len(vol_cols):
            idx_end = df.columns.get_loc(vol_cols[i + 1])

        # 🔹 Solo colonne numeriche come metriche (esclude 'Struttura')
        metric_cols_full = [c for c in df.columns[idx_start:idx_end] if pd.api.types.is_numeric_dtype(df[c])]

        # 🔹 Estrai nome metrica dalle parentesi
        metric_cols = []
        metric_to_col = {}
        for col in metric_cols_full:
            if "(" in col and ")" in col:
                metric_name = col.split("(")[1].split(")")[0].strip()
            else:
                metric_name = col.strip()
            metric_cols.append(metric_name)
            metric_to_col[metric_name] = col

        metric_map[struttura_nome] = metric_cols
        metric_column_map[struttura_nome] = metric_to_col

    st.info(f"Strutture trovate: {list(metric_map.keys())}")
    for s, m in metric_map.items():
        st.write(f"**{s}** -> metriche: {m}")

    # ============================================================
    # 5) IDENTIFICAZIONE VECCHIO VS NUOVO
    # ============================================================
    df["TipoPiano"] = df[col_plan].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

    # ============================================================
    # 6) COSTRUZIONE RISULTATI MULTI-STRUTTURA E MULTI-METRICA
    # ============================================================
    results = []

    for id_val in df[col_id].unique():
        for struct, metrics in metric_map.items():
            subset = df[df["Struttura"] == struct]
            subset = subset[subset[col_id] == id_val]
            if subset.empty:
                continue
            metric_to_col = metric_column_map[struct]
            for metric in metrics:
                old_row = subset[subset["TipoPiano"] == "Vecchio"]
                new_row = subset[subset["TipoPiano"] == "Nuovo"]
                if old_row.empty or new_row.empty:
                    continue
                old_value = old_row.iloc[0][metric_to_col[metric]]
                new_value = new_row.iloc[0][metric_to_col[metric]]
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

    if results_df.empty:
        st.warning("⚠️ Nessun dato valido trovato per il confronto Vecchio/Nuovo.")
        st.stop()

    st.subheader("📊 Risultati del confronto")
    st.dataframe(results_df)

    # ============================================================
    # 7) RADAR PLOT
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
    # 8) HEATMAP
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
    # 9) WILCOXON
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
    # 10) EXPORT EXCEL
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
    # 11) RISULTATO FINALE
    # ============================================================
    st.subheader("🏆 RISULTATO FINALE")
    summary = results_df["Migliore"].value_counts()
    st.write(summary)
    if summary.get("Nuovo", 0) > summary.get("Vecchio", 0):
        st.success("🎉 Il nuovo modello RapidPlan risulta migliore!")
    else:
        st.error("⚠️ Il vecchio modello sembra migliore… per questi dati.")
