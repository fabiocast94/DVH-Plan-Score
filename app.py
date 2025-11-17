import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from rapidfuzz import process, fuzz
import json, os
from io import BytesIO

st.set_page_config(page_title="Dose Hunter", layout="wide")

# ============================================================
# 0) CRITERI METRICHE
# ============================================================

METRIC_CRITERIA = {
    "HI": "lower", "D95": "higher", "D98": "higher", "D2": "lower",
    "D50": "lower", "Dmax": "lower", "Dmean": "lower",
    "V95": "higher", "V90": "higher", "V107": "lower",
    "V20": "lower", "V5": "lower", "V10": "lower",
    "CI": "higher"
}

EQUIV_THRESHOLD = 0.01

def better_value(old, new, metric):
    if pd.isna(old) or pd.isna(new):
        return "N/A"
    crit = METRIC_CRITERIA.get(metric, "lower")
    rel = abs(new - old) / old if old != 0 else 0
    if rel < EQUIV_THRESHOLD:
        return "Equivalente"
    if crit == "lower":
        return "Nuovo" if new < old else "Vecchio"
    else:
        return "Nuovo" if new > old else "Vecchio"


# ============================================================
# 1) PRESET DISTRETTI
# ============================================================

BASE_PRESET = {
    "Prostata": {
        "PTV": ["ptv", "prostata", "boost"],
        "Retto": ["retto", "rect"],
        "Vescica": ["vescica", "bladder"],
        "Femori": ["fem", "femur"]
    },

    "Testa-Collo": {
        "PTV": ["ptv"],
        "Midollo": ["midollo", "spinal"],
        "ParotideDx": ["parotid dx", "parotide dx", "parotid right"],
        "ParotideSx": ["parotid sx", "parotide sx", "parotid left"]
    },

    "Torace": {
        "PTV": ["ptv"],
        "PolmoneDx": ["lung dx", "polmone dx", "right lung"],
        "PolmoneSx": ["lung sx", "polmone sx", "left lung"],
        "Cuore": ["cuore", "heart"],
        "Esofago": ["esofago", "esophagus"]
    },

    "Breast": {
        "PTV": ["ptv", "breast", "mamma", "mammella", "seno"],
        "PolmoneIpsilaterale": ["lung ipsi", "polmone ipsi"],
        "PolmoneControlaterale": ["lung contro", "polmone contro"],
        "Cuore": ["cuore", "heart"],
        "MammellaControlaterale": ["breast contra", "mammella contro", "seno contro"]
    }
}


# ============================================================
# 2) LOAD / SAVE PRESET CUSTOM
# ============================================================

def load_custom_preset():
    if os.path.exists("preset_custom.json"):
        with open("preset_custom.json", "r") as f:
            return json.load(f)
    return {}

def save_custom_preset(preset):
    with open("preset_custom.json", "w") as f:
        json.dump(preset, f, indent=4)


# MERGE preset custom con quelli base
PRESET_DISTRETTI = BASE_PRESET.copy()
custom = load_custom_preset()
for dist, items in custom.items():
    if dist not in PRESET_DISTRETTI:
        PRESET_DISTRETTI[dist] = items
    else:
        for k, v in items.items():
            if k in PRESET_DISTRETTI[dist]:
                PRESET_DISTRETTI[dist][k] += v
            else:
                PRESET_DISTRETTI[dist][k] = v


# ============================================================
# 3) FUNZIONI DI RICONOSCIMENTO STRUTTURE
# ============================================================

def autodetect_distretto(volumes):
    checks = {
        "Breast": ["breast", "mamma", "seno"],
        "Prostata": ["retto", "vescica", "prostata"],
        "Testa-Collo": ["parotid", "midollo", "spinal"],
        "Torace": ["polmone", "lung", "cuore", "esofago"]
    }
    for vol in volumes:
        name = vol.lower()
        for dist, keys in checks.items():
            if any(k in name for k in keys):
                return dist
    return list(PRESET_DISTRETTI.keys())[0]


def riconosci_struttura_fuzzy(nome, preset):
    nome_low = nome.lower()
    alias_lookup = []

    for struttura, alias_list in preset.items():
        for alias in alias_list:
            alias_lookup.append((struttura, alias))

    if not alias_lookup:
        return None

    matches = process.extract(
        nome_low,
        [a[1] for a in alias_lookup],
        scorer=fuzz.partial_ratio,
        limit=1
    )

    best_alias = matches[0]
    score = best_alias[1]
    alias_value = best_alias[0]

    if score > 70:
        for struttura, alias in alias_lookup:
            if alias == alias_value:
                return struttura

    return None


def detect_side(structure):
    s = structure.lower()
    if "dx" in s or "right" in s or "_r" in s:
        return "Dx"
    if "sx" in s or "left" in s or "_l" in s:
        return "Sx"
    return None


# ============================================================
# 4) INTERFACCIA PRINCIPALE
# ============================================================

st.title("🔬 Analisi Dose Hunter – Versione Avanzata")

uploaded_file = st.file_uploader("Carica file Excel Dose Hunter", type=["xlsx"])

if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# Identificazione colonne ID / plan
col_id = [c for c in df.columns if "id" in c.lower()][0]
col_plan = [c for c in df.columns if "plan" in c.lower()][0]
vol_cols = [c for c in df.columns if "(vol)" in c.lower()]

# ============================================================
# 5) AUTO-DETECT DISTRETTO
# ============================================================

st.subheader("🏷️ Identificazione automatica strutture")

dist_auto = autodetect_distretto(vol_cols)
st.write(f"🔍 Distretto rilevato: **{dist_auto}**")

distretto = st.selectbox("Distretto", list(PRESET_DISTRETTI.keys()),
                         index=list(PRESET_DISTRETTI.keys()).index(dist_auto))
preset = PRESET_DISTRETTI[distretto]


# ============================================================
# 6) MAPPATURA STRUTTURE
# ============================================================

df["Struttura"] = None
manual_struct_map = {}

for vol in vol_cols:
    guessed = riconosci_struttura_fuzzy(vol, preset)
    side = detect_side(vol)

    if guessed and side and "Dx" in guessed or "Sx" in guessed:
        pass

    strutture = ["(auto)"] + list(preset.keys())

    choice = st.selectbox(
        f"Struttura per '{vol}'",
        strutture,
        index=strutture.index(guessed) if guessed in strutture else 0
    )

    final = guessed if choice == "(auto)" else choice
    manual_struct_map[vol] = final

    if final not in preset:
        if st.checkbox(f"Aggiungi '{final}' al preset del distretto?"):
            PRESET_DISTRETTI[distretto][final] = [final.lower()]
            save_custom_preset(PRESET_DISTRETTI)

# Applica al dataframe
for vol_col, struct in manual_struct_map.items():
    df.loc[df[vol_col].notna(), "Struttura"] = struct


# ============================================================
# 7) COSTRUZIONE METRIC MAP
# ============================================================

metric_map = {}
metric_column_map = {}

for i, vol in enumerate(vol_cols):
    struct = manual_struct_map[vol]

    start = df.columns.get_loc(vol) + 1
    end = df.columns.get_loc(vol_cols[i + 1]) if i + 1 < len(vol_cols) else df.shape[1]

    metrics = []
    m_to_c = {}

    for col in df.columns[start:end]:
        if not pd.api.types.is_numeric_dtype(df[col]): continue
        name = col.split("(")[1].split(")")[0] if "(" in col else col
        metrics.append(name)
        m_to_c[name] = col

    metric_map[struct] = metrics
    metric_column_map[struct] = m_to_c


# ============================================================
# 8) ANALISI NUOVO vs VECCHIO
# ============================================================

df["TipoPiano"] = df[col_plan].apply(lambda x:
    "Nuovo" if "new" in str(x).lower() else "Vecchio"
)

results = []

for id_val in df[col_id].unique():
    temp = df[df[col_id] == id_val]

    for struct, metrics in metric_map.items():
        sub = temp[temp["Struttura"] == struct]
        if sub.empty: continue

        for m in metrics:
            col = metric_column_map[struct][m]

            v_old = sub[sub["TipoPiano"] == "Vecchio"][col].iloc[0]
            v_new = sub[sub["TipoPiano"] == "Nuovo"][col].iloc[0]

            results.append({
                "ID": id_val,
                "Struttura": struct,
                "Metrica": m,
                "Valore Vecchio": v_old,
                "Valore Nuovo": v_new,
                "Δ %": (v_new - v_old) / v_old * 100 if v_old != 0 else 0,
                "Migliore": better_value(v_old, v_new, m)
            })

results_df = pd.DataFrame(results)


# ============================================================
# 9) FILTRI
# ============================================================

st.sidebar.header("Filtri")

struct_sel = st.sidebar.multiselect("Strutture", results_df["Struttura"].unique())
metric_sel = st.sidebar.multiselect("Metriche", results_df["Metrica"].unique())

filtered = results_df.copy()
if struct_sel:
    filtered = filtered[filtered["Struttura"].isin(struct_sel)]
if metric_sel:
    filtered = filtered[filtered["Metrica"].isin(metric_sel)]


# ============================================================
# 10) PTV vs OAR
# ============================================================

PTV_df = filtered[filtered["Struttura"].str.contains("PTV", case=False)]
OAR_df = filtered[~filtered["Struttura"].str.contains("PTV", case=False)]

st.subheader("📊 PTV")
st.dataframe(PTV_df)

st.subheader("🫁 OAR")
st.dataframe(OAR_df)


# ============================================================
# 11) WILCOXON
# ============================================================

wilcox = []
for struct in filtered["Struttura"].unique():
    for met in filtered["Metrica"].unique():
        vals = filtered[(filtered["Struttura"] == struct) &
                        (filtered["Metrica"] == met)]

        if len(vals) < 2:
            continue

        try:
            stat, p = wilcoxon(vals["Valore Vecchio"], vals["Valore Nuovo"])
        except:
            stat, p = None, None

        wilcox.append([struct, met, stat, p])

wilcox_df = pd.DataFrame(wilcox, columns=["Struttura", "Metrica", "Stat", "p-value"])
wilcox_df["Significativo"] = wilcox_df["p-value"] < 0.05

st.subheader("🧪 Test Wilcoxon")
st.dataframe(wilcox_df)


# ============================================================
# 12) EXPORT
# ============================================================

output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    PTV_df.to_excel(writer, "PTV", index=False)
    OAR_df.to_excel(writer, "OAR", index=False)
    wilcox_df.to_excel(writer, "Wilcoxon", index=False)

st.download_button(
    "📥 Scarica Excel",
    data=output.getvalue(),
    file_name="Analisi_DoseHunter.xlsx"
)


# ============================================================
# 13) RISULTATO FINALE
# ============================================================

st.subheader("🏁 Risultato finale")

summary = filtered["Migliore"].value_counts()
st.write(summary)

if summary.get("Nuovo", 0) > summary.get("Vecchio", 0):
    st.success("🎉 Il nuovo modello è complessivamente migliore!")
else:
    st.error("⚠️ Il vecchio modello prevale su queste metriche.")
