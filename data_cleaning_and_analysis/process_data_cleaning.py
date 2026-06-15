import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Eingabedateien
# --------------------------------------------------
DATEIEN_LISTE = [
    Path(r"../data/lstm ready data/M200310_processdata_2024_2026.csv"),

]

# --------------------------------------------------
# Output
# --------------------------------------------------
OUT_DIR = Path(r"../data/lstm ready data/")
OUT_BASENAME = "M200310_processdata_bereinigt_2024_206_gesamt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Hilfsfunktion: Leere Werte erkennen
# --------------------------------------------------
def is_empty(series):
    return (
        series.isna()
        | (series.astype(str).str.strip() == "")
        | (series.astype(str).str.lower().str.strip() == "null")
    )

# --------------------------------------------------
# Zeitfenster
# --------------------------------------------------
intervals = [
    ("04:45:00", "06:00:00"),
    ("06:00:00", "07:00:00"),
    ("07:00:00", "08:00:00"),
    ("08:00:00", "09:00:00"),
    ("09:30:00", "10:00:00"),
    ("10:00:00", "11:00:00"),
    ("11:00:00", "12:00:00"),
    ("12:00:00", "13:15:00"),
    ("13:15:00", "14:00:00"),
    ("14:00:00", "15:00:00"),
    ("15:00:00", "16:00:00"),
    ("16:00:00", "17:00:00"),
    ("17:00:00", "17:30:00"),
    ("17:30:00", "18:00:00"),
    ("18:00:00", "19:00:00"),
    ("19:00:00", "20:00:00"),
    ("20:00:00", "21:00:00"),
    ("21:00:00", "21:45:00"),
    ("21:45:00", "22:00:00"),
    ("22:00:00", "23:00:00"),
    ("23:00:00", "00:00:00"),
    ("00:00:00", "01:00:00"),
    ("01:00:00", "01:30:00"),
    ("01:30:00", "02:00:00"),
    ("02:00:00", "03:00:00"),
    ("03:00:00", "04:00:00"),
    ("04:00:00", "04:45:00"),
]

def find_interval(time_str):
    if pd.isna(time_str):
        return pd.Series([np.nan, np.nan])

    try:
        t = pd.to_datetime(str(time_str)).time()

        for start, end in intervals:
            start_t = pd.to_datetime(start).time()
            end_t = pd.to_datetime(end).time()

            if start_t < end_t:
                if start_t <= t < end_t:
                    return pd.Series([start, end])
            else:
                if t >= start_t or t < end_t:
                    return pd.Series([start, end])

        return pd.Series([np.nan, np.nan])

    except Exception:
        return pd.Series([np.nan, np.nan])

# --------------------------------------------------
# Alle Dateien verarbeiten
# --------------------------------------------------
dfs = []

for file in DATEIEN_LISTE:
    print(f"Lade Datei: {file}")

    df = pd.read_csv(
        file,
        sep=";",
        dtype=str,
        low_memory=False
    )

    # --------------------------------------------------
    # 1. Werte übernehmen
    # --------------------------------------------------

    mask = is_empty(df["Rückgem. Gutmenge (GMEIN)"])
    df.loc[mask, "Rückgem. Gutmenge (GMEIN)"] = df.loc[
        mask, "Rückgem. Gutmenge (MEINH)"
    ]

    mask = is_empty(df["Basismengeneinheit (=GMEIN)"])
    df.loc[mask, "Basismengeneinheit (=GMEIN)"] = df.loc[
        mask, "Mengeneinheit Vrg. (=MEINH)"
    ]

    mask = is_empty(df["Istendzt.Durchf."])
    df.loc[mask, "Istendzt.Durchf."] = df.loc[
        mask, "Ist-Ende Vorg."
    ]

    mask = is_empty(df["Ende Durchf.(Zeit)"])
    df.loc[mask, "Ende Durchf.(Zeit)"] = df.loc[
        mask, "Istendzt.Durchf."
    ]
    # Ist-Ende Vorg. -> Ende Durchf.(Dat.)
    mask = is_empty(df["Ende Durchf.(Dat.)"])
    df.loc[mask, "Ende Durchf.(Dat.)"] = df.loc[
        mask, "Ist-Ende Vorg."
    ]

    # --------------------------------------------------
    # Zeitfenster zuweisen
    # --------------------------------------------------
    df[["zeit_von", "zeit_bis"]] = df["Ende Durchf.(Zeit)"].apply(find_interval)

    dfs.append(df)

# --------------------------------------------------
# Zusammenführen
# --------------------------------------------------
df_gesamt = pd.concat(dfs, ignore_index=True)
# --------------------------------------------------
# --------------------------------------------------
# Nach dem concat aller Dateien
# --------------------------------------------------
df_gesamt["Vorgangsmenge (MEINH)"] = pd.to_numeric(
    df_gesamt["Vorgangsmenge (MEINH)"],
    errors="coerce"
).fillna(0)

df_gesamt["stückzahl_je_zeitraum"] = (
    df_gesamt.groupby(
        ["Ende Durchf.(Dat.)", "zeit_von", "zeit_bis"]
    )["Vorgangsmenge (MEINH)"]
    .transform("sum")
)
# --------------------------------------------------
# Speichern
# --------------------------------------------------
output_file = OUT_DIR / f"{OUT_BASENAME}.csv"

df_gesamt.to_csv(
    output_file,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(f"Datei gespeichert: {output_file}")