# parameters.py

# ==========================================
# Dateipfad
# ==========================================
FILE_PATH_FAULTS = r"C:\Users\louis\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\stoerliste_mit_auftragsdaten_2024_2026.csv"

# ==========================================
# Training Configuration
# ==========================================
SEQ_LENGTH = 24
lstm_1_units = 96
dropout_1 = 0.3
lstm_2_units = 32
dropout_2 = 0.2
dense_units = 16
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 22
TEST_SPLIT = 0.20
Failure_Class_Multiplier = 1.5
L2_RATE = 0.01

# -------------------------------------------------------------------------
# THRESHOLD-EXPERIMENTE
# -------------------------------------------------------------------------
# hier einen festen Wert (z. B. 0.30 oder 0.50), um manuell zu testen.
# auf None, um den mathematisch optimalen Youden-Index zu berechnen und zu nutzen.
MANUAL_THRESHOLD = None
# -------------------------------------------------------------------------

# ZIELVARIABLEN
TARGETS = [
    'Dauer Anlagen-Ausfall',
    'Dauer Anlagen-Ausfall intern',
    'Störung aufgrund Vormaterial',
    'Störfall'
]

# NUMERISCHE FEATURES
NUMERIC_FEATURES = [
    'Wochentag',
    'Dauer Arbeitszeit',
    'Menge i. O. L4',
    'MengeGesamtNIO',

    # Permutation Ergebnis: folgende Features werden raus genommen
    # 'Jahr', 'Monat', 'KW', 'Tag',
    # 'Zeit_von_min', 'Avg_Time_To_Failure_min', 'last_fail_time',
    # 'Zeit_seit_letztem_Fehler_min', 'Anzahl MA', 'Anzahl Störfälle Zeitfenster',
    # 'Stoerfall_7d_mean', 'Stoerfall_30d_mean', 'Dauer Logistik- Defizite', 'Zeit_bis_min'
]

# KATEGORIALE FEATURES
CATEGORICAL_FEATURES = [
    'Wochenende (J/N)',
    'Materialnummer'
]
# ==========================================
# Target und Datum
# ==========================================
TARGET_DURATION = 'Dauer Anlagen-Ausfall'

DT_PROCESS_DATE = 'Ende Durchf.(Dat.)'
DT_FAULTS = 'Datum'