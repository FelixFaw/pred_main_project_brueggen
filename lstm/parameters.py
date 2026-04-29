# parameters.py

# --- File Configuration ---
FILE_PATH_FAULTS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\processed\aufschreibung_mta_clean.xlsx"
FILE_PATH_PROCESS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\raw\M200310 - STW_MT_Anlage.XLSX"

# --- Training Configuration ---
SEQ_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.000001
TEST_SPLIT = 0.1

# --- Target Definition (What to predict) ---
# (Diese Spalten dürfen KEINE Features werden, sonst schummeln wir)
TARGET_DURATION = '"Dauer\nAnlagen-Ausfall\n"'
TARGET_STATION = 'Station/ OP'

# --- Datetime Columns for joining ---
DT_FAULTS = 'DatumNEU'
DT_PROCESS_DATE = 'Istenddat.Durchf.'
DT_PROCESS_TIME = 'Istendzt.Durchf.'