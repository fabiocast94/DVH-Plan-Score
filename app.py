import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from io import BytesIO
import base64
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1) CRITERI METRICHE
# ============================================================
METRIC_CRITERIA = {
    "HI": "lower", "D95": "higher", "D98": "higher", "D2": "lower",
    "D50": "lower", "Dmax": "lower", "Dmean": "lower",
    "V95": "higher","V90": "higher","V107": "lower",
    "V20": "lower","V5": "lower","V10": "lower",
    "CI": "higher","PI95":"higher","CI95":"higher",
    "Mean":"lower","Max":"lower","Min":"lower"
}

# Aggiunta criterio MU (meno MU = piano più efficiente)
METRIC_CRITERIA["MU"] = "lower"

EQUIV_THRESHOLD = 0.01  # soglia 1%

def better_value(old, new, metric):
    if pd.isna(old) or pd.isna(new): 
        return "N/A"
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
logo_path = Path("06_Humanitas.png")
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
st.title("🔬 Analisi Multi-Struttura, Multi-Metrica e Monitor Units (MU)")

uploaded_file = st.file_uploader("Carica file Excel o CSV", type=["xlsx", "csv"])

def load_any_file(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        try:
            sample = file.read(2048).decode("utf-8", errors="ignore")
            file.seek(0)

            if ";" in sample and sample.count(";") > sample.count(","):
                sep = ";"
            elif "\t" in sample:
                sep = "\t"
            else:
                sep = ","

            try:
                return pd.read_csv(file, sep=sep, encoding="utf-8")
            except:
                file.seek(0)
                return pd.read_csv(file, sep=sep, encoding="ISO-8859-1")

        except Exception as e:
            st.error(f"Errore nel caricamento del CSV: {e}")
            return None

    return pd.read_excel(file)

# ======== LETTURA ============
if uploaded_file:
    df = load_any_file(uploaded_file)
    if df is None:
        st.stop()

    df.columns = df.columns.str.strip()

    # ============================================================
    # CHECK MU
    # ============================================================
    if "MU" not in df.columns:
        st.warning("⚠️ Nessuna colonna 'MU' trovata. L’analisi MU sarà saltata.")
        has_MU = False
    else:
        has_MU = True

    # ============================================================
    # IDENTIFICAZIONE STRUTTURE
    # ============================================================
    structures = {}
    for col in df.columns:
        if "(" in col and ")" in col:
            struct = col.split("(")[0].strip()
            metric = col.split("(")[1].split(")")[0].strip()
            structures.setdefault(struct, {})[metric] = col

    st.write("Strutture trovate:", list(structures.keys()))

    # ============================================================
    # Tipo Piano Nuovo vs Vecchio
    # ============================================================
    plan_cols = [c for c in df.columns if "plan" in c.lower()]
    col_plan = plan_cols[0] if plan_cols else "planID"

    id_cols = [c for c in df.columns if "id" in c.lower()]
    col_id = id_cols[0] if id_cols else "patientID"

    df["TipoPiano"] = df[col_plan].apply(
        lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio"
    )

    # ============================================================
    # Creazione risultati (metriche strutture)
    # ============================================================
    results = []
    for id_val in df[col_id].unique():
        temp = df[df[col_id] == id_val]

        for struct, metrics in structures.items():
            for met, col in metrics.items():
                v_old = temp[temp["TipoPiano"]=="Vecchio"][col].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
                v_new = temp[temp["TipoPiano"]=="Nuovo"][col].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan

                winner = better_value(v_old, v_new, met)
                diff_pct = ((v_new - v_old)/v_old*100 if v_old and not pd.isna(v_old) else np.nan)

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
    # ANALISI MU
    # ============================================================
    if has_MU:
        MU_results = []
        for id_val in df[col_id].unique():
            temp = df[df[col_id] == id_val]

            MU_old = temp[temp["TipoPiano"]=="Vecchio"]["MU"].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
            MU_new = temp[temp["TipoPiano"]=="Nuovo"]["MU"].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan

            diff_pct = ((MU_new - MU_old) / MU_old * 100) if (MU_old and not pd.isna(MU_old)) else np.nan
            winner = better_value(MU_old, MU_new, "MU")

            MU_results.append({
                "ID": id_val,
                "MU Vecchio": MU_old,
                "MU Nuovo": MU_new,
                "Δ% MU": diff_pct,
                "Piano più efficiente (MU)": winner
            })

        MU_df = pd.DataFrame(MU_results)

        # Efficienza MU/Gy se presente Dose
        if "Dose" in df.columns:
            df["MU_per_Gy"] = df["MU"] / df["Dose"]

            MU_eff_list = []
            for id_val in df[col_id].unique():
                temp = df[df[col_id] == id_val]

                eff_old = temp[temp["TipoPiano"]=="Vecchio"]["MU_per_Gy"].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
                eff_new = temp[temp["TipoPiano"]=="Nuovo"]["MU_per_Gy"].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan

                diff_pct = ((eff_new - eff_old)/eff_old*100) if (eff_old and not pd.isna(eff_old)) else np.nan
                winner = better_value(eff_old, eff_new, "MU")

                MU_eff_list.append({
                    "ID": id_val,
                    "MU/Gy Vecchio": eff_old,
                    "MU/Gy Nuovo": eff_new,
                    "Δ% MU/Gy": diff_pct,
                    "Piano più efficiente (MU/Gy)": winner
                })

            MU_eff_df = pd.DataFrame(MU_eff_list)
        else:
            MU_eff_df = pd.DataFrame()

    # ============================================================
    # Sidebar Filtri
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
    # TABELLE MU
    # ============================================================
    if has_MU:
        st.subheader("⚡ Monitor Units (MU)")
        st.dataframe(MU_df)

        if not MU_eff_df.empty:
            st.subheader("⚡ Efficienza MU per Gy")
            st.dataframe(MU_eff_df)

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

    # Aggiunta Wilcoxon MU
    if has_MU:
        try:
            stat_MU, p_MU = wilcoxon(MU_df["MU Vecchio"], MU_df["MU Nuovo"])
        except:
            stat_MU, p_MU = None, None

        wilcox_df = pd.concat([
            wilcox_df,
            pd.DataFrame({
                "Struttura": ["GLOBAL"],
                "Metrica": ["MU"],
                "Statistic": [stat_MU],
                "p-value": [p_MU],
                "Significativo": [p_MU < 0.05 if p_MU is not None else False]
            })
        ], ignore_index=True)

    st.subheader("📊 Risultati Wilcoxon")
    st.dataframe(wilcox_df)

    # ============================================================
    # Heatmap per struttura
    # ============================================================
    st.subheader("🔥 Heatmap per ogni struttura")

    for struct in results_filtered["Struttura"].unique():
        df_struct = results_filtered[results_filtered["Struttura"] == struct]

        if df_struct.empty:
            continue

        heatmap_data = df_struct.pivot_table(
            index="ID",
            columns="Metrica",
            values="Δ %"
        )

        fig_width = max(6, len(heatmap_data.columns)*1.5)
        fig_height = max(3, len(heatmap_data.index)*1.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Δ % (Nuovo vs Vecchio)"}
        )
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Heatmap Δ% – {struct}")
        plt.tight_layout()
        st.pyplot(fig)

    # Heatmap MU
    if has_MU:
        st.subheader("🔥 Heatmap MU (Δ% Nuovo vs Vecchio)")
        heatmap_mu = MU_df.pivot_table(index="ID", values="Δ% MU")
        fig, ax = plt.subplots(figsize=(4, len(heatmap_mu)*1.2))
        sns.heatmap(heatmap_mu, cmap="coolwarm", center=0, annot=True, fmt=".1f")
        st.pyplot(fig)

    # ============================================================
    # Download Excel
    # ============================================================
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        PTV_df.to_excel(writer,"PTV",index=False)
        OAR_df.to_excel(writer,"OAR",index=False)
        wilcox_df.to_excel(writer,"Wilcoxon",index=False)

        if has_MU:
            MU_df.to_excel(writer,"MU",index=False)
            if not MU_eff_df.empty:
                MU_eff_df.to_excel(writer,"MU_eff",index=False)

    st.download_button(
        "📥 Scarica Excel completo",
        data = output.getvalue(),
        file_name="Analisi_RapidPlan_Advanced.xlsx"
    )

    # ============================================================
    # RISULTATO FINALE
    # ============================================================
    st.subheader("🏁 RISULTATO FINALE")
    summary = results_filtered["Migliore"].value_counts()
    st.write(summary)

    if summary.get("Nuovo",0) > summary.get("Vecchio",0):
        st.success("🎉 Il nuovo modello RapidPlan risulta complessivamente migliore!")
    else:
        st.error("⚠️ Il modello vecchio sembra migliore su queste metriche.")
