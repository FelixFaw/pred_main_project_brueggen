# parameters.py

# Pfad zu den Prozessdaten (Datei 1)
FILE_PATH_PROCESS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\M200310_processdata_2024_2026.csv"
# Pfad zu den MTA-Aufschreibungen / Ausfällen (Datei 2)
FILE_PATH_FAULTS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\lstm ready data\aufschreibung_mta_clean_gesamt.csv"

# --- Training Configuration ---
#num_features = 14
#SEQ_LENGTH = 12
#lstm_1_units = 16
#dropout_1 = 0.2
#lstm_2_units = 64
#dropout_2 = 0.4
#dense_units = 48
#LEARNING_RATE = 0.00519
#BATCH_SIZE = 128
#NEG_WEIGHT = 0.2

#EPOCHS = 10
#TEST_SPLIT = 0.20
#POS_WEIGHT = 1.0


# --- Training Configuration (Optimiert für stabile Konvergenz) ---
# --- Training Configuration (Vergrößert für komplexe Zeitreihenmuster) ---
num_features = 14
SEQ_LENGTH = 12
lstm_1_units = 32       # Zurückgefahren für weniger Overfitting-Kapazität
dropout_1 = 0.3         # Erhöht
lstm_2_units = 64       # Zurückgefahren
dropout_2 = 0.3         # Erhöht
dense_units = 32
LEARNING_RATE = 0.001  # Niedrigere Lernrate für extrem stabiles, sanftes Sinken des Loss
BATCH_SIZE = 32
EPOCHS = 10
TEST_SPLIT = 0.20
best_threshold = 0.32
# --- Target Definition (What to predict) ---
TARGET_DURATION = 'Dauer Anlagen-Ausfall' # Datei 2
TARGET_STATION = 'Station/ OP'

# --- Datetime Columns for joining ---
DT_PROCESS_DATE = 'Ende Durchf.(Dat.)'  # Datumsspalte aus Datei 1
DT_FAULTS = 'DatumNEU'                  # Datumsspalte aus Datei 2