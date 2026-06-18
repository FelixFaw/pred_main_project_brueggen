# parameters.py

# ==========================================
# Dateipfade (Relativ für bessere Portabilität)
# ==========================================
FILE_PATH_PROCESS = r"C:\Users\louis\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\M200310_processdata_bereinigt_2024_206_gesamt.csv"
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
# Setze hier einen festen Wert (z. B. 0.30 oder 0.50), um manuell zu testen.
# Setze es auf None, um den mathematisch optimalen Youden-Index zu nutzen.
MANUAL_THRESHOLD = None
# -------------------------------------------------------------------------

# ZIELVARIABLEN (Targets)
TARGETS = [
    'Dauer Anlagen-Ausfall',
    'Dauer Anlagen-Ausfall intern',
    'Störung aufgrund Vormaterial',
    'Störfall'
]

# NUMERISCHE FEATURES
NUMERIC_FEATURES = [
    # --- Die echten Prozess-Treiber ---
    'Wochentag',  # Rhythmus der Woche (z.B. Montags-Anlauf-Probleme)
    #'Zeit_bis_min',  # Wo in der Schicht/im Auftrag befinden wir uns?
    'Dauer Arbeitszeit',  # Wie lange läuft die Maschine schon?
    'Menge i. O. L4',  # Konkreter Output
    'MengeGesamtNIO',  # Ausschuss (sehr wichtiges Warnsignal für Verschleiß!)

    # --- Wir werfen makro-zeitliches Rauschen und Overfitting-Fallen RAUS ---
    # 'Jahr', 'Monat', 'KW', 'Tag',
    # 'Zeit_von_min', 'Avg_Time_To_Failure_min', 'last_fail_time',
    # 'Zeit_seit_letztem_Fehler_min', 'Anzahl MA', 'Anzahl Störfälle Zeitfenster',
    # 'Stoerfall_7d_mean', 'Stoerfall_30d_mean', 'Dauer Logistik- Defizite'
]

# KATEGORIALE FEATURES
CATEGORICAL_FEATURES = [
    # --- Kontext-Features ---
    'Wochenende (J/N)',
    'Materialnummer'  # Wir geben ihr noch eine Chance, da sie ohne das "Jahr" vielleicht besser wirkt
]
# ==========================================
# Einzelne Target Definitionen & Datumsspalten
# ==========================================
TARGET_DURATION = 'Dauer Anlagen-Ausfall'
TARGET_STATION = 'Station/ OP'

DT_PROCESS_DATE = 'Ende Durchf.(Dat.)'
DT_FAULTS = 'Datum'