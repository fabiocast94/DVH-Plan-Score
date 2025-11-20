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
    "Mean":"lower","Max":"lower","Min":"lower",
    "MU":"lower"
}

EQUIV_THRESHOLD = 0.01  # soglia 1%

def better_value(old, new, metric):
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    crit = METRIC_CRITERIA.get(metric, "lower")
    rel_diff = abs(new - old) / old if old != 0 else 0
    if rel_diff < EQUIV_THRESHOLD:
        return "Equivalente"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"

# =============================
# LOGO
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
    st.warning("⚠️ Logo 06_Humanitas non trovato.")

# ============================================================
st.title("🔬 Analisi Multi-Struttura, Multi-Metrica e Monitor Units (MU)")

uploaded_file = st.file_uploader("Carica file Excel o CSV", type=["xlsx", "csv"])

def load_any_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            sample = file.read(2048).decode("utf-8", errors="ignore")
            file.seek(0)
            sep = "," if sample.count(",") >= sample.count(";") else ";"
            if "\t" in sample: sep = "\t"
            try:
                return pd.read_csv(file, sep=sep, encoding="utf-8")
            except:
                file.seek(0)
                return pd.read_csv(file, sep=sep, encoding="ISO-8859-1")
        except Exception as e:
            st.error(f"Errore CSV: {e}")
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
    has_MU = "MU" in df.columns
    if not has_MU:
        st.warning("⚠️ Nessuna colonna 'MU' trovata. L’analisi MU sarà esclusa.")

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
    # RISULTATI METRICHE STRUTTURE
    # ============================================================
    results = []
    for id_val in df[col_id].unique():
        temp = df[df[col_id] == id_val]
        for struct, metrics in structures.items():
            for met, col in metrics.items():
                v_old = temp[temp["TipoPiano"] == "Vecchio"][col].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
                v_new = temp[temp["TipoPiano"] == "Nuovo"][col].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan
                diff_pct = ((v_new - v_old) / v_old * 100) if (v_old and not pd.isna(v_old)) else np.nan
                winner = better_value(v_old, v_new, met)
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
    # MU ANALYSIS
    # ============================================================
    if has_MU:
        MU_list = []
        for id_val in df[col_id].unique():
            temp = df[df[col_id]==id_val]
            MU_old = temp[temp["TipoPiano"]=="Vecchio"]["MU"].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
            MU_new = temp[temp["TipoPiano"]=="Nuovo"]["MU"].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan
            diff_pct = ((MU_new - MU_old)/MU_old*100) if (MU_old and not pd.isna(MU_old)) else np.nan
            winner = better_value(MU_old, MU_new, "MU")
            MU_list.append({"ID": id_val,"MU Vecchio":MU_old,"MU Nuovo":MU_new,"Δ% MU":diff_pct,"Piano più efficiente (MU)":winner})

        MU_df = pd.DataFrame(MU_list)
        MU_summary_df = pd.DataFrame({
            "Valore": ["Media","DevStd"],
            "MU Vecchio": [MU_df["MU Vecchio"].mean(), MU_df["MU Vecchio"].std()],
            "MU Nuovo": [MU_df["MU Nuovo"].mean(), MU_df["MU Nuovo"].std()]
        })

        if "Dose" in df.columns:
            df["MU_per_Gy"] = df["MU"] / df["Dose"]
            MU_eff_list = []
            for id_val in df[col_id].unique():
                temp = df[df[col_id]=="Vecchio"]
                eff_old = temp[temp["TipoPiano"]=="Vecchio"]["MU_per_Gy"].iloc[0] if not temp[temp["TipoPiano"]=="Vecchio"].empty else np.nan
                eff_new = temp[temp["TipoPiano"]=="Nuovo"]["MU_per_Gy"].iloc[0] if not temp[temp["TipoPiano"]=="Nuovo"].empty else np.nan
                diff_pct = ((eff_new - eff_old)/eff_old*100) if (eff_old and not pd.isna(eff_old)) else np.nan
                winner = better_value(eff_old, eff_new, "MU")
                MU_eff_list.append({"ID":id_val,"MU/Gy Vecchio":eff_old,"MU/Gy Nuovo":eff_new,"Δ% MU/Gy":diff_pct,"Piano più efficiente (MU/Gy)":winner})
            MU_eff_df = pd.DataFrame(MU_eff_list)
            MU_Gy_summary_df = pd.DataFrame({
                "Valore":["Media","DevStd"],
                "MU/Gy Vecchio":[MU_eff_df["MU/Gy Vecchio"].mean(), MU_eff_df["MU/Gy Vecchio"].std()],
                "MU/Gy Nuovo":[MU_eff_df["MU/Gy Nuovo"].mean(), MU_eff_df["MU/Gy Nuovo"].std()]
            })
        else:
            MU_eff_df = pd.DataFrame()
            MU_Gy_summary_df = pd.DataFrame()

    # ============================================================
    # SIDEBAR FILTRI
    # ============================================================
    st.sidebar.header("🔍 Filtri")
    structs_sel_upper = [s.upper() for s in st.sidebar.multiselect(
        "Seleziona strutture",
        results_df["Struttura"].unique()
    )]
    metrics_sel = st.sidebar.multiselect(
        "Seleziona metriche",
        results_df["Metrica"].unique()
    )
    results_filtered = results_df.copy()
    if structs_sel_upper:
        results_filtered = results_filtered[results_filtered["Struttura_upper"].isin(structs_sel_upper)]
    if metrics_sel:
        results_filtered = results_filtered[results_filtered["Metrica"].isin(metrics_sel)]

    # ============================================================
    # PTV / OAR
    # ============================================================
    PTV_df = results_filtered[results_filtered["Struttura_upper"].str.contains("PTV")]
    OAR_df = results_filtered[~results_filtered["Struttura_upper"].str.contains("PTV")]

    st.subheader("📊 Risultati PTV")
    st.dataframe(PTV_df)

    st.subheader("🫁 Risultati OAR")
    st.dataframe(OAR_df)

# ============================================================
# STATISTICHE METRICHE PTV / OAR
# ============================================================
def summary_stats(df_group):
    # Raggruppa per Struttura e Metrica
    return df_group.groupby(["Struttura", "Metrica"]).agg(
        Media=('Δ %', 'mean'),
        DevStd=('Δ %', 'std')
    ).reset_index()

PTV_summary = summary_stats(PTV_df)
OAR_summary = summary_stats(OAR_df)

st.subheader("📈 Statistiche Δ% PTV (Media & Dev Std)")
st.dataframe(PTV_summary)

st.subheader("📈 Statistiche Δ% OAR (Media & Dev Std)")
st.dataframe(OAR_summary)


    # ============================================================
    # WILCOXON
    # ============================================================
    wilcox = []
    for struct in results_filtered["Struttura"].unique():
        for met in results_filtered["Metrica"].unique():
            vals = results_filtered[(results_filtered["Struttura"]==struct)&(results_filtered["Metrica"]==met)]
            if len(vals)<2: continue
            try:
                stat,p = wilcoxon(vals["Valore Vecchio"], vals["Valore Nuovo"])
            except:
                stat,p = None,None
            wilcox.append([struct, met, stat, p])
    wilcox_df = pd.DataFrame(wilcox, columns=["Struttura","Metrica","Statistic","p-value"])
    wilcox_df["Significativo"] = wilcox_df["p-value"] < 0.05

    if has_MU:
        try:
            stat_MU, p_MU = wilcoxon(MU_df["MU Vecchio"], MU_df["MU Nuovo"])
        except:
            stat_MU, p_MU = None, None
        wilcox_df.loc[len(wilcox_df)] = ["GLOBAL","MU",stat_MU,p_MU,p_MU<0.05 if p_MU is not None else False]

    st.subheader("📊 Risultati Wilcoxon")
    st.dataframe(wilcox_df)

    # ============================================================
    # HEATMAP PER STRUTTURA
    # ============================================================
    st.subheader("🔥 Heatmap per ogni struttura")
    for struct in results_filtered["Struttura"].unique():
        df_struct = results_filtered[results_filtered["Struttura"]==struct]
        heatmap_data = df_struct.pivot_table(index="ID", columns="Metrica", values="Δ %")
        if heatmap_data.empty or heatmap_data.isna().all().all():
            st.info(f"ℹ️ Nessun dato disponibile per la heatmap della struttura **{struct}**.")
            continue
        fig,ax = plt.subplots(figsize=(max(6,len(heatmap_data.columns)*1.5),max(3,len(heatmap_data.index)*1.2)))
        sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=True, fmt=".1f", linewidths=0.5,
                    cbar_kws={"label":"Δ % (Nuovo vs Vecchio)"})
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Heatmap Δ% – {struct}")
        st.pyplot(fig)

    # ============================================================
    # DOWNLOAD EXCEL
    # ============================================================
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        PTV_df.to_excel(writer,"PTV",index=False)
        OAR_df.to_excel(writer,"OAR",index=False)
        wilcox_df.to_excel(writer,"Wilcoxon",index=False)
        PTV_summary.to_excel(writer,"PTV_stats",index=False)
        OAR_summary.to_excel(writer,"OAR_stats",index=False)
        if has_MU:
            MU_df.to_excel(writer,"MU",index=False)
            MU_summary_df.to_excel(writer,"MU_stats",index=False)
            if not MU_eff_df.empty:
                MU_eff_df.to_excel(writer,"MU_eff",index=False)
                MU_Gy_summary_df.to_excel(writer,"MU_Gy_stats",index=False)

    st.download_button(
        "📥 Scarica Excel completo",
        data=output.getvalue(),
        file_name="Analisi_RapidPlan_Advanced.xlsx"
    )

    # ============================================================
    # RISULTATO FINALE
    # ============================================================
    st.subheader("🏁 RISULTATO FINALE")
    summary = results_filtered["Migliore"].value_counts()
    st.write(summary)
    if summary.get("Nuovo",0) > summary.get("Vecchio",0):
        st.success("🎉 Il modello RapidPlan NUOVO è complessivamente migliore!")
    else:
        st.error("⚠️ Il modello RapidPlan VECCHIO sembra migliore su queste metriche.")
