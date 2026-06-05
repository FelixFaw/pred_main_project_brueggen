import pandas as pd

# Dateien laden
df1 = pd.read_excel(
    r"C:\Users\golde\PycharmProjects\pred_main_project_brueggen\data\raw\mta2024to2026\M200310 STW-Mittelteilanlage 2024_2025.XLSX"
)

df2 = pd.read_excel(
    r"C:\Users\golde\PycharmProjects\pred_main_project_brueggen\data\raw\M200310 STW-Mittelteilanlage 2026.XLSX"
)

# Zusammenführen
merged_df = pd.concat([df1, df2], ignore_index=True)

# Datums-Spalte umwandeln
merged_df["Ist-Ende Vorg."] = pd.to_datetime(
    merged_df["Ist-Ende Vorg."],
    errors="coerce"
)

# Nach Datum sortieren
merged_df = merged_df.sort_values(by="Ist-Ende Vorg.")

# Excel speichern
merged_df.to_excel(
    r"C:\Users\golde\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\M200310_processdata_2024_2026.xlsx",
    index=False
)

# CSV speichern
merged_df.to_csv(
    r"C:\Users\golde\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\M200310_processdata_2024_2026.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print("Excel- und CSV-Datei erfolgreich erstellt.")