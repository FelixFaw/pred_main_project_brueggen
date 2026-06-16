# Data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Daten laden
# KORREKTUR: .xlsx Dateien müssen mit read_excel eingelesen werden
file_path = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\aufschreibung_mta_clean_gesamt_2024to2026.xlsx"
df = pd.read_excel(file_path)

# 2. Features und Targets definieren
# Wir trennen numerische Features (für den Scaler) und kategorische Features
numeric_features = [
'Datum', 'KW', 'Jahr', 'Monat', 'Tag', 'Wochentag', 'Quartal', 'Zeit von', 'Zeit bis',
'Dauer Arbeitszeit', 'Anzahl MA', 'Menge N.i. O.', 'Menge i. O. L4', 'Menge i. O. L5', 'MengeGesamtNIO',
'Dauer Org-Mangel', 'Dauer Anlagen-Ausfall', 'Störung aufgrund Vormaterial', 'Dauer Anlagen-Ausfall intern',
'Dauer Logistik- Defizite', 'Anzahl/ Std.', 'Sollzeit/ Stück (Min)', 'Zeit_von_min', 'Zeit_bis_min',
 'Anzahl Störfälle Zeitfenster', 'Störfall'
]

categorical_features = ['Schicht','Station/ OP_raw', 'Station/ OP_2', 'Station/ OP_1', 'Bemerkung_norm',]

# Namen der Zielspalten in deiner Excel (bitte bei Bedarf exakt abgleichen!)
col_failure_duration = 'Dauer Anlagen-Ausfall'
col_station = 'Station/ OP'
col_reason = 'Störung aufgrund Vormaterial'

# 3. Vorbereitung der Features
print("Bereinige numerische Spalten von falschen Datentypen (z.B. Datumsangaben)...")
# KUGELSICHER: Wir zwingen alle numerischen Spalten, echte Zahlen zu sein.
# Alles, was Text oder ein Datum ist (wie '2023-04-20'), wird zu NaN und dann sofort zu 0.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Kategoriale Daten in Zahlen umwandeln (One-Hot Encoding für das LSTM)
df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

# Kombiniere numerische Features und die neuen Encodings
features_to_scale = pd.concat([df[numeric_features], df_encoded], axis=1)

feature_cols = [c for c in df.columns if c not in ['Datum', 'Dauer_Anlagen_Ausfall', 'Zeit_von_min', 'Zeit_bis_min']]

# 4. MinMaxScaler anwenden (Nur auf Features!)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features_to_scale)

# Zurück in ein DataFrame wandeln
df_features_final = pd.DataFrame(scaled_data, columns=features_to_scale.columns)

# 5. Targets (Zielvariablen) vorbereiten
# Diese werden NICHT skaliert, da es Klassen/Labels sind.
print("Bereite Zielvariablen (Targets) vor...")

# Output 1: WHEN (Binär: Gab es einen Ausfall?)
df_features_final['is_failure'] = (df[col_failure_duration] > 0).astype(int)

# Output 2: WHERE (Welche Station? - Encodiert als Zahl)
station_le = LabelEncoder()
df_features_final['station_encoded'] = station_le.fit_transform(df[col_station].astype(str))

# Output 3: WHY (Fehlergrund? - Encodiert als Zahl)
reason_le = LabelEncoder()
df_features_final['reason_encoded'] = reason_le.fit_transform(df[col_reason].astype(str))

# 6. Ergebnisse prüfen & Speichern
print("\n--- Zusammenfassung der Präparation ---")
print(f"Originale Zeilen: {len(df)}")
print(f"Features nach Encoding: {len(df_features_final.columns) - 3}") # -3 wegen der Targets
print(f"Gefundene Stationen: {len(station_le.classes_)}")
print(f"Gefundene Fehlergründe: {len(reason_le.classes_)}")

# Speichern der fertigen Daten für die main.py
output_path = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\processed\lstm_ready_data.csv"
df_features_final.to_csv(output_path, index=False)

print(f"\nDatei erfolgreich gespeichert unter:\n{output_path}")