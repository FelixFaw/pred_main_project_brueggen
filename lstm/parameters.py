# parameters.py

# --- File Configuration ---
FILE_PATH_FAULTS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\raw\Daten\aufschreibung_mta_clean_2024to2026.csv"


# --- Training Configuration ---
num_features = 14
SEQ_LENGTH = 12
lstm_1_units = 16
dropout_1 = 0.2
lstm_2_units = 64
dropout_2 = 0.4
dense_units = 48
LEARNING_RATE = 0.00519
BATCH_SIZE = 128
NEG_WEIGHT = 1.25

EPOCHS = 25
TEST_SPLIT = 0.1
POS_WEIGHT = 1.0


# --- Target Definition (What to predict) ---
# (Diese Spalten dürfen KEINE Features werden, sonst schummeln wir)
TARGET_DURATION = '"Dauer\nAnlagen-Ausfall\n"'
TARGET_STATION = 'Station/ OP'

# --- Datetime Columns for joining ---
DT_FAULTS = 'DatumNEU'
DT_PROCESS_DATE = 'Istenddat.Durchf.'
DT_PROCESS_TIME = 'Istendzt.Durchf.'