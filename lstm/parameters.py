# parameters.py

# --- File Configuration ---
FILE_PATH_FAULTS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\raw\Störliste STW-Mittelteilanlage 2023_NEU.xlsx"
FILE_PATH_PROCESS = r"C:\Users\tanne\PycharmProjects\pred_main_project_brueggen\data\raw\M200310 - STW_MT_Anlage.XLSX"

# --- Training Configuration ---
SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2

# --- Target Definition (What to predict) ---
# (Diese drei Spalten dürfen KEINE Features werden, sonst schummeln wir)
TARGET_DURATION = '"Dauer\nAnlagen-Ausfall\n"'
TARGET_STATION = 'Station/ OP'
TARGET_REASON = 'Fehlercode'

# --- Datetime Columns for joining ---
DT_FAULTS = 'DatumNEU'
DT_PROCESS_DATE = 'Istenddat.Durchf.'
DT_PROCESS_TIME = 'Istendzt.Durchf.'