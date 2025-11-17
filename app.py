import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# =========================================================
# ⚙️ 1) CRITERI MIGLIORAMENTO PER METRICA
# =========================================================
METRIC_CRITERIA = {
    "HI": "lower",
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
    "CI": "higher",
}

def better_value(old, new, metric):
    crit = METRIC_CRITERIA.get(metric, "lower")
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    return "Nuovo" if new > old else "Vecchio"

# =========================================================
# 🧠 2) CLASSIFICATORE STRUTTURE
# =========================================================
def classify_structure(name):
    name_low = name.lower()

    if any(x in name_low for x in ["ptv", "ctv", "gtv", "boost"]):
        return "PTV"

    if "breast" in name_low or "mamma" in name_low:
        return "BREAST"

    if any(x in name_low for x in ["lens", "lung", "heart", "kidney", "rectum",
                                   "bladder", "esophagus", "trachea", "liver",
                                   "spinal", "optic", "chiasm", "parotid"]):
        return "OAR"

    return "OAR"

# =========================================================
# 🚀 3) STREAMLIT UI
# =========================================================
st.title("🧠 Analisi RapidPlan – Multi-Struttura")

uploaded_file = st.file_uploader("Carica file Dose Hunter (.xlsx)", type=["xlsx"])

if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)
st.write("📁 Anteprima:")
st.dataframe(df.head())

# =========================================================
# 🔍 4) IDENTIFICAZIONE COLONNE
# =========================================================
id_col = [c for c in df.columns if "id" in c.lower()][0]
plan_col = [c for c in df.columns if "plan" in c.lower()][0]
vol_cols = [c for c in df.columns if "(vol)" in c.lower()]

df["TipoPiano"] = df[plan_col].apply(lambda x: "Nuovo" if "new" in str(x).lower() else "Vecchio")

metric_map = {}
metric_column_map = {}
structure_assignment = {}

for i, vol in enumerate(vol_cols):
    struct = vol.replace("(vol)", "").strip()
    class_type = classify_structure(struct)
    structure_assignment[struct] = class_type

    idx_start = df.columns.get_loc(vol) + 1
    idx_end = df.columns.get_loc(vol_cols[i+1]) if i+1 < len(vol_cols) else df.shape[1]

    metric_cols = [c for c in df.columns[idx_start:idx_end] if pd.api.types.is_numeric_dtype(df[c])]
    metric_clean = {}
    clean_names = []

    for m in metric_cols:
        if "(" in m:
            name = m.split("(")[1].split(")")[0]
        else:
            name = m
        clean_names.append(name)
        metric_clean[name] = m

    metric_map[struct] = clean_names
    metric_column_map[struct] = metric_clean

# =========================================================
# 🧮 5) COSTRUZIONE RISULTATI
# =========================================================
results = []

for pid in df[id_col].unique():
    for struct in metric_map:
        subset = df[df[id_col] == pid]

        for metric in metric_map[struct]:
            col = metric_column_map[struct][metric]

            old = subset[subset["TipoPiano"] == "Vecchio"]
            new = subset[subset["TipoPiano"] == "Nuovo"]
            if old.empty or new.empty:
                continue

            oldv = old.iloc[0][col]
            newv = new.iloc[0][col]
            better = better_value(oldv, newv, metric)

            results.append({
                "ID": pid,
                "Struttura": struct,
                "Metrica": metric,
                "Classe": structure_assignment[struct],
                "Old": oldv,
                "New": newv,
                "Migliore": better
            })

results_df = pd.DataFrame(results)
st.success(f"📊 RISULTATI GENERATI ({len(results_df)} confronti)")

# =========================================================
# 📈 6) RADAR PLOT
# =========================================================
st.subheader("📈 Radar Plot ")

sel_id = st.selectbox("Seleziona ID", results_df["ID"].unique())
sel_struct = st.selectbox("Seleziona Struttura", results_df["Struttura"].unique())

rad = results_df[(results_df["ID"] == sel_id) & (results_df["Struttura"] == sel_struct)]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=rad["Old"], theta=rad["Metrica"], fill='toself', name="Vecchio"))
fig.add_trace(go.Scatterpolar(r=rad["New"], theta=rad["Metrica"], fill='toself', name="Nuovo"))

st.plotly_chart(fig)

# =========================================================
# 🔥 7) HEATMAP PTV & OAR
# =========================================================
def make_heatmap(class_type, title):

    pvt = results_df[results_df["Classe"] == class_type]
    if pvt.empty:
        st.warning(f"Nessun dato per {title}")
        return

    pivot = pvt.pivot_table(index="ID", columns=["Struttura", "Metrica"], values="Migliore",
                            aggfunc=lambda x: x.iloc[0])
    mat = pivot.replace({"Vecchio": 0, "Nuovo": 1})

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(mat, annot=pivot, cmap="coolwarm", fmt="", ax=ax)
    st.write(f"🔥 Heatmap {title}")
    st.pyplot(fig)
    return pivot

ptv_pivot = make_heatmap("PTV", "PTV")
oar_pivot = make_heatmap("OAR", "OAR")

# =========================================================
# 📉 8) WILCOXON
# =========================================================
wilc = []
for struct in results_df["Struttura"].unique():
    for metric in results_df["Metrica"].unique():
        vals = results_df[(results_df["Struttura"] == struct) &
                          (results_df["Metrica"] == metric)]
        if len(vals) < 2: continue
        try:
            w, p = wilcoxon(vals["Old"], vals["New"])
        except:
            w, p = None, None

        wilc.append([struct, metric, w, p])

wilcox_df = pd.DataFrame(wilc, columns=["Struttura", "Metrica", "Statistic", "p-value"])
st.subheader("📊 Wilcoxon Test")
st.dataframe(wilcox_df)

# =========================================================
# 📥 9) EXPORT EXCEL
# =========================================================
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as wr:
    results_df.to_excel(wr, sheet_name="Risultati", index=False)
    if ptv_pivot is not None: ptv_pivot.to_excel(wr, sheet_name="PTV")
    if oar_pivot is not None: oar_pivot.to_excel(wr, sheet_name="OAR")
    wilcox_df.to_excel(wr, sheet_name="Wilcoxon", index=False)

st.download_button("📥 Scarica Excel", output.getvalue(), "Analisi_RapidPlan.xlsx")

# =========================================================
# 🏁 10) RISULTATO FINALE
# =========================================================
st.subheader("🏆 RISULTATO COMPLESSIVO")

count = results_df["Migliore"].value_counts()
st.write(count)

if count.get("Nuovo", 0) > count.get("Vecchio", 0):
    st.success("🎉 Il nuovo modello RapidPlan risulta migliore!")
else:
    st.error("⚠️ Il modello precedente risulta globalmente migliore.")
