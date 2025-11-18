import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from io import BytesIO
import base64
from pathlib import Path

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
EQUIV_THRESHOLD = 0.01  # soglia 1%

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

# =============================
# LOGO FISSO CENTRATO IN ALTO
# =============================
logo_path = Path("06_Humanitas.png")  # stesso nome del tuo file

if logo_path.exists():
    encoded_logo = base64.b64encode(logo_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{encoded_logo}" 
                 style="width: 400px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Logo 06_Humanitas non trovato nella cartella dello script.")

# ============================================================

st.title("🔬 Analisi Multi-Struttura e Multi-Metrica")

uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ============================================================
    # IDENTIFICAZIONE STRUTTURE BASATA SUI PREFISSI DELLE COLONNE
    # ============================================================
    structures = {}
    for col in df.columns:
        if "(" in col and ")" in col:
            struct = col.split("(")[0].strip()
            metric = col.split("(")[1].split(")")[0].strip()
            if struct not in structures:
                structures[struct] = {}
            structures[struct][metric] = col

    st.write("Strutture trovate:", list(structures.keys()))

    # ============================================================
    # Tipo Piano Nuovo vs Vecchio
    # ============================================================
    plan_cols = [c for c in df.columns if "plan" in c.lower()]
    col_plan = plan_cols[0] if plan_cols else "planID"
    id_cols = [c for c in df.columns if "id" in c.lower()]
    col_id = id_cols[0] if id_cols else "patientID"

    df["TipoPiano"] = df[col_plan].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

    # ============================================================
    # Creazione risultati
    # ============================================================
    results = []
    for id_val in df[col_id].unique():
        temp = df[df[col_id] == id_val]

        for struct, metrics in structures.items():
            for met, col in metrics.items():
                v_old = temp[temp["TipoPiano"]=="Vecchio"][col].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
                v_new = temp[temp["TipoPiano"]=="Nuovo"][col].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan

                winner = better_value(v_old, v_new, met)
                diff_pct = ((v_new - v_old)/v_old*100 if v_old and not pd.isna(v_old) else 0)

                results.append({
                    "ID": id_val,
                    "Struttura": struct,
                    "Metrica": met,
                    "Valore Vecchio": v_old,
                    "Valore Nuovo": v_new,
                    "Δ %": diff_pct,
                    "Migliore": winner
                })

    results_df = pd.DataFrame(results)
    results_df["Struttura_upper"] = results_df["Struttura"].str.upper()

    # ============================================================
    # Sidebar Filtri (SOLO SELEZIONE MANUALE)
    # ============================================================
    st.sidebar.header("🔍 Filtri")

    structs_sel_upper = [s.upper() for s in st.sidebar.multiselect(
        "Seleziona strutture",
        results_df["Struttura"].unique(),
        default=None
    )]

    metrics_sel = st.sidebar.multiselect(
        "Seleziona metriche",
        results_df["Metrica"].unique(),
        default=None
    )

    results_filtered = results_df.copy()

    if structs_sel_upper:
        results_filtered = results_filtered[results_filtered["Struttura_upper"].isin(structs_sel_upper)]
    if metrics_sel:
        results_filtered = results_filtered[results_filtered["Metrica"].isin(metrics_sel)]

    # ============================================================
    # Separazione PTV vs OAR
    # ============================================================
    PTV_df = results_filtered[results_filtered["Struttura"].str.upper().str.contains("PTV")]
    OAR_df = results_filtered[~results_filtered["Struttura"].str.upper().str.contains("PTV")]

    st.subheader("📊 Risultati PTV")
    st.dataframe(PTV_df)

    st.subheader("🫁 Risultati OAR")
    st.dataframe(OAR_df)

    # ============================================================
    # Test di Wilcoxon
    # ============================================================
    wilcox = []
    for struct in results_filtered["Struttura"].unique():
        for met in results_filtered["Metrica"].unique():
            vals = results_filtered[(results_filtered["Struttura"]==struct)&(results_filtered["Metrica"]==met)]
            if len(vals) < 2: continue
            try:
                stat, p = wilcoxon(vals["Valore Vecchio"], vals["Valore Nuovo"])
            except:
                stat, p = None, None
            wilcox.append([struct, met, stat, p])

    wilcox_df = pd.DataFrame(wilcox, columns=["Struttura","Metrica","Statistic","p-value"])
    wilcox_df["Significativo"] = wilcox_df["p-value"] < 0.05

    show_only_sig = st.sidebar.checkbox("Mostra solo metriche significative (p < 0.05)")
    if show_only_sig:
        results_filtered = results_filtered.merge(
            wilcox_df[wilcox_df["Significativo"]], on=["Struttura","Metrica"]
        )

    st.subheader("📊 Risultati Wilcoxon")
    st.dataframe(wilcox_df)

    # ============================================================
    # Download Excel
    # ============================================================
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
    # ============================================================
    st.subheader("🏁 RISULTATO FINALE")
    summary = results_filtered["Migliore"].value_counts()
    st.write(summary)

    if summary.get("Nuovo",0) > summary.get("Vecchio",0):
        st.success("🎉 Il nuovo modello RapidPlan risulta complessivamente migliore!")
    else:
        st.error("⚠️ Il modello vecchio sembra migliore su queste metriche.")
