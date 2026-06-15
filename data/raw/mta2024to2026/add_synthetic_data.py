import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

# ============================================================
# CSV laden
# ============================================================

df = pd.read_csv(
    "aufschreibung_mta_clean_gesamt.csv",
    sep=";",
    encoding="utf-8"
)

# ============================================================
# ZEITSTEMPEL ERZEUGEN
#
# Datum + "Zeit von" werden zu einem echten Timestamp
# zusammengeführt.
#
# Falls dein Datumsformat anders ist:
# dayfirst=True anpassen.
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["Datum"].astype(str) + " " + df["Zeit von"].astype(str),
    dayfirst=True,
    errors="coerce"
)

# Nach Zeit sortieren
df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# STUNDEN SEIT SCHICHTBEGINN
#
# Frühschicht = 04:45
# Spätschicht = 13:15
#
# Für unbekannte Schichten wird NaN gesetzt.
# ============================================================

def get_shift_start(row):

    schicht = str(row["Schicht"]).lower()

    datum = row["timestamp"].date()

    if "früh" in schicht:
        return pd.Timestamp(f"{datum} 04:45:00")

    elif "spät" in schicht:
        return pd.Timestamp(f"{datum} 13:15:00")

    return pd.NaT


df["shift_start"] = df.apply(get_shift_start, axis=1)

df["stunden_seit_schichtbeginn"] = (
    (df["timestamp"] - df["shift_start"])
    .dt.total_seconds()
    / 3600
)

# ============================================================
# WOCHENENDE
#
# Samstag = 5
# Sonntag = 6
# ============================================================

df["wochenende"] = (
    df["timestamp"].dt.dayofweek >= 5
).astype(int)

# ============================================================
# MONATSANFANG
#
# Erste 5 Kalendertage des Monats
# ============================================================

df["monatsanfang"] = (
    df["timestamp"].dt.day <= 5
).astype(int)

# ============================================================
# MONATSENDE
#
# Letzte 5 Kalendertage des Monats
# ============================================================

letzter_tag = (
    df["timestamp"] + MonthEnd(0)
).dt.day

df["monatsende"] = (
    (letzter_tag - df["timestamp"].dt.day) < 5
).astype(int)

# ============================================================
# ZEIT SEIT LETZTEM FEHLER GLOBAL
#
# Abstand zur vorherigen Fehlerzeile
# in Stunden.
# ============================================================

df["zeit_seit_letztem_fehler_h"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
    / 3600
)

# ============================================================
# WARTUNGEN ERKENNEN
#
# Sucht nach typischen Wartungsbegriffen
# in der Bemerkungsspalte.
#
# Spaltennamen ggf. anpassen.
# ============================================================

wartung_pattern = (
    "wartung|instandhaltung|service"
)

df["is_wartung"] = (
    df["Bemerkung_std"]
    .fillna("")
    .str.lower()
    .str.contains(wartung_pattern, regex=True)
)

# Zeitpunkte aller Wartungen
wartungen = df.loc[
    df["is_wartung"],
    "timestamp"
]

# ============================================================
# ZEIT SEIT LETZTER WARTUNG
#
# Für jede Zeile wird die letzte bekannte
# Wartung gesucht.
# ============================================================

df["letzte_wartung"] = np.nan

letzte_wartung = pd.NaT

for idx in df.index:

    if df.loc[idx, "is_wartung"]:
        letzte_wartung = df.loc[idx, "timestamp"]

    df.loc[idx, "letzte_wartung"] = letzte_wartung

df["letzte_wartung"] = pd.to_datetime(
    df["letzte_wartung"]
)

df["zeit_seit_letzter_wartung_h"] = (
    (
        df["timestamp"]
        - df["letzte_wartung"]
    )
    .dt.total_seconds()
    / 3600
)

# ============================================================
# STATIONSBEZOGENE FEATURES
#
# Gruppierung nach Station/OP
# ============================================================

station_col = "Station/ OP"

# ------------------------------------------------------------
# ZEIT BIS ZUM NÄCHSTEN FEHLER DER GLEICHEN STATION
# ------------------------------------------------------------

df["next_failure_station"] = (
    df.groupby(station_col)["timestamp"]
      .shift(-1)
)

df["time_to_next_failure_station_h"] = (
    (
        df["next_failure_station"]
        - df["timestamp"]
    )
    .dt.total_seconds()
    / 3600
)

# ------------------------------------------------------------
# DURCHSCHNITTLICHE ZEIT ZWISCHEN AUSFÄLLEN
# JE STATION
# ------------------------------------------------------------

df["prev_failure_station"] = (
    df.groupby(station_col)["timestamp"]
      .shift(1)
)

df["failure_interval_h"] = (
    (
        df["timestamp"]
        - df["prev_failure_station"]
    )
    .dt.total_seconds()
    / 3600
)

df["avg_failure_interval_station_h"] = (
    df.groupby(station_col)["failure_interval_h"]
      .expanding()
      .mean()
      .reset_index(level=0, drop=True)
)

# ============================================================
# PRODUKTIONSFEATURES
#
# Mehrere Fehler können in derselben Taktung auftreten.
#
# Deshalb zählen wir Mengen nur einmal
# je Zeitstempel.
# ============================================================

taktung_df = (
    df
    .sort_values("timestamp")
    .drop_duplicates(
        subset=["timestamp"]
    )
    .copy()
)

# ============================================================
# PRODUKTE SEIT LETZTEM AUSFALL
#
# Da die Datei ausschließlich Fehler enthält,
# entspricht dies der kumulierten Menge
# über die eindeutigen Fehler-Taktungen.
# ============================================================

taktung_df["produkte_seit_letztem_ausfall"] = (
    taktung_df["MengeGesamt"]
    .cumsum()
)

# ============================================================
# NIO SEIT LETZTEM AUSFALL
# ============================================================

taktung_df["nio_seit_letztem_ausfall"] = (
    taktung_df["Menge N.I. O."]
    .cumsum()
)

# Zurück auf Originaldaten mergen
df = df.merge(
    taktung_df[
        [
            "timestamp",
            "produkte_seit_letztem_ausfall",
            "nio_seit_letztem_ausfall"
        ]
    ],
    on="timestamp",
    how="left"
)

# ============================================================
# ROLLING FEHLERHÄUFIGKEIT
#
# Anzahl Fehler in den letzten
# 7 bzw. 30 Tagen.
# ============================================================

df = df.set_index("timestamp")

df["rolling_failures_7d"] = (
    pd.Series(
        1,
        index=df.index
    )
    .rolling("7D")
    .sum()
)

df["rolling_failures_30d"] = (
    pd.Series(
        1,
        index=df.index
    )
    .rolling("30D")
    .sum()
)

df = df.reset_index()

# ============================================================
# AUFRÄUMEN
# ============================================================

drop_cols = [
    "shift_start",
    "next_failure_station",
    "prev_failure_station",
    "failure_interval_h",
    "letzte_wartung"
]

df = df.drop(
    columns=[c for c in drop_cols if c in df.columns]
)

# ============================================================
# SPEICHERN
# ============================================================

df.to_csv(
    "aufschreibung_mta_mit_features.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("Feature Engineering abgeschlossen.")