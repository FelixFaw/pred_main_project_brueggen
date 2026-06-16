# parameters.py

# ==========================================
# Dateipfade (Relativ für bessere Portabilität)
# ==========================================
# Pfad zu den Prozessdaten (Datei 1)
FILE_PATH_PROCESS = r"C:\Users\louis\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\M200310_processdata_bereinigt_2024_206_gesamt.csv"
# Pfad zu den MTA-Aufschreibungen / Ausfällen (Datei 2)
FILE_PATH_FAULTS = r"C:\Users\louis\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\stoerliste_mit_auftragsdaten_2024_2026.csv"

# ==========================================
# Training Configuration
# ==========================================
num_features = 14  # Hinweis: Wird durch One-Hot-Encoding später im Code meist dynamisch überschrieben
SEQ_LENGTH = 24
lstm_1_units = 32
dropout_1 = 0.3
lstm_2_units = 64
dropout_2 = 0.3
dense_units = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 22
TEST_SPLIT = 0.20
best_threshold = 0.549


# ZIELVARIABLEN (Targets)
TARGETS = [
    'Dauer Anlagen-Ausfall',
    'Dauer Anlagen-Ausfall intern',
    'Störung aufgrund Vormaterial',
    'Störfall'
]

# NUMERISCHE FEATURES
NUMERIC_FEATURES = [
    'Jahr', 'Quartal', 'Monat', 'KW', 'Wochentag', 'Tag',
    'Zeit_von_min', 'Zeit_bis_min',
    'Dauer Arbeitszeit', 'Anzahl MA', 'Menge N.i. O.', 'Menge i. O. L4',
    'Menge i. O. L5', 'MengeGesamtNIO', 'Anzahl/ Std.', 'Sollzeit/ Stück (Min)', 'Stoerfall_7d_mean', 'Stoerfall_30d_mean', 'Avg_Time_To_Failure_min', 'last_fail_time' 
    'Dauer Org-Mangel', 'Dauer Logistik- Defizite', 'Anzahl Störfälle Zeitfenster', 'Zeit_seit_letztem_Fehler_min'
]

# KATEGORIALE FEATURES
#  via One-Hot-Encoding umgewandelt
CATEGORICAL_FEATURES = [
    'Wochenende (J/N)',
    'Schicht',
    'Materialnummer',
    #'Kundenauftragsnummer'
   # 'Station/ OP_raw',
  #  'Station/ OP_1',
   # 'Station/ OP_2',
    #'Bemerkung_norm' damit kommen wir auf 1571 Features... ohne ist ca 1450 Features weniger
]

# ==========================================
# Einzelne Target Definitionen & Datumsspalten
# ==========================================
TARGET_DURATION = 'Dauer Anlagen-Ausfall' # Datei 2
TARGET_STATION = 'Station/ OP'

DT_PROCESS_DATE = 'Ende Durchf.(Dat.)'    # Datumsspalte aus Datei 1
DT_FAULTS = 'Datum'                       # Datumsspalte aus Datei 2