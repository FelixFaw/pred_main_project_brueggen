# parameters.py

# --- File Configuration ---
FILE_PATH_FAULTS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\processed\aufschreibung_mta_clean.xlsx"
FILE_PATH_PROCESS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\processed\lstm_ready_data.csv"

# --- Training Configuration ---
SEQ_LENGTH = 12
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.00519
TEST_SPLIT = 0.1

# --- Target Definition (What to predict) ---
# (Diese Spalten dürfen KEINE Features werden, sonst schummeln wir)
TARGET_DURATION = '"Dauer\nAnlagen-Ausfall\n"'
TARGET_STATION = 'Station/ OP'

# --- Datetime Columns for joining ---
DT_FAULTS = 'DatumNEU'
DT_PROCESS_DATE = 'Istenddat.Durchf.'
DT_PROCESS_TIME = 'Istendzt.Durchf.'