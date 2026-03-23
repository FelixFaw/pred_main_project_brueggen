# helping_functions.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import parameters as p
from collections import defaultdict


def load_and_preprocess_data_all_features():
    print(f"Lade Störliste: {p.FILE_PATH_FAULTS}")
    df_faults = pd.read_excel(p.FILE_PATH_FAULTS, engine='openpyxl')

    print(f"Lade Prozessdaten: {p.FILE_PATH_PROCESS}")
    df_process = pd.read_excel(p.FILE_PATH_PROCESS, engine='openpyxl')

    # --- 1. Spaltennamen bereinigen ---
    df_faults.columns = df_faults.columns.str.replace('\n', ' ').str.replace('\xa0', ' ').str.replace('"',
                                                                                                      '').str.strip()
    df_process.columns = df_process.columns.str.replace('\n', ' ').str.replace('\xa0', ' ').str.replace('"',
                                                                                                        '').str.strip()

    target_duration_clean = p.TARGET_DURATION.replace('\n', ' ').replace('\xa0', ' ').replace('"', '').strip()
    target_station_clean = p.TARGET_STATION.replace('\n', ' ').replace('\xa0', ' ').replace('"', '').strip()
    target_reason_clean = p.TARGET_REASON.replace('\n', ' ').replace('\xa0', ' ').replace('"', '').strip()

    # --- 2. Zeitstempel für den Merge erstellen ---
    print("Synchronisiere Zeitstempel beider Dateien...")
    df_faults['Join_Time'] = pd.to_datetime(df_faults[p.DT_FAULTS], errors='coerce')

    if p.DT_PROCESS_DATE in df_process.columns and p.DT_PROCESS_TIME in df_process.columns:
        df_process['Join_Time'] = pd.to_datetime(
            df_process[p.DT_PROCESS_DATE].astype(str).str.split(' ').str[0] + ' ' +
            df_process[p.DT_PROCESS_TIME].astype(str),
            errors='coerce'
        )
    else:
        df_process['Join_Time'] = pd.to_datetime(df_process[p.DT_PROCESS_DATE], errors='coerce')

    df_faults.dropna(subset=['Join_Time'], inplace=True)
    df_process.dropna(subset=['Join_Time'], inplace=True)
    df_faults.sort_values('Join_Time', inplace=True)
    df_process.sort_values('Join_Time', inplace=True)

    # --- 3. MERGE (Verschmelzung) ---
    df = pd.merge_asof(df_faults, df_process, on='Join_Time', direction='backward')
    print(f"Merge abgeschlossen. Roh-Spaltenanzahl gesamt: {len(df.columns)}")

    required_targets = [target_duration_clean, target_station_clean, target_reason_clean]
    for target in required_targets:
        if target not in df.columns:
            raise KeyError(f"Spalte {target} nicht gefunden. Bitte passe den Namen in parameters.py an.")

    # Extrahieren von Zeit-Features
    df['Time_Hour'] = df['Join_Time'].dt.hour
    df['Time_DayOfWeek'] = df['Join_Time'].dt.dayofweek

    # --- 4. Targets erstellen ---
    df['is_failure'] = (pd.to_numeric(df[target_duration_clean], errors='coerce').fillna(0) > 0).astype(int)

    df[target_station_clean] = df[target_station_clean].astype(str).replace('0', 'No_Station').replace('nan',
                                                                                                       'No_Station')
    df[target_reason_clean] = df[target_reason_clean].fillna('No_Error').astype(str).replace('0', 'No_Error').replace(
        'nan', 'No_Error')

    station_encoder = LabelEncoder()
    reason_encoder = LabelEncoder()
    df['station_encoded'] = station_encoder.fit_transform(df[target_station_clean])
    df['reason_encoded'] = reason_encoder.fit_transform(df[target_reason_clean])

    # --- 5. DYNAMISCHES FEATURE ENGINEERING (Kugelsicher) ---
    print("Analysiere und transformiere alle verbleibenden Spalten...")
    excluded = [
        'is_failure', 'station_encoded', 'reason_encoded',
        target_duration_clean, target_station_clean, target_reason_clean,
        'Join_Time', p.DT_FAULTS, p.DT_PROCESS_DATE, p.DT_PROCESS_TIME,
        'Datum', 'Zeit von', 'Zeit bis', 'Unnamed: 36'
    ]

    feature_cols = []
    encoders = defaultdict(LabelEncoder)
    numeric_cols_for_scaling = []

    for col in df.columns:
        if col in excluded:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # KUGELSICHERER CHECK: Ist es mathematisch als Zahl zu verstehen?
        if pd.api.types.is_numeric_dtype(df[col]):
            # Es ist eine saubere Zahl (Int, Float, Bool)
            df[col] = df[col].fillna(0)
            feature_cols.append(col)
            numeric_cols_for_scaling.append(col)
        else:
            # Es ist Text (String, Object, Category) - wie das 't'
            # Wir versuchen deutsche Kommas zu Punkten zu machen, um versteckte Zahlen zu retten
            temp_str = df[col].astype(str).str.replace(',', '.').str.strip()
            temp_numeric = pd.to_numeric(temp_str, errors='coerce')

            # Wenn mehr als 50% danach kaputt (NaN) sind, ist es echter Text (wie bei 'Log')
            if temp_numeric.isna().sum() > (len(df) * 0.5):
                df[col] = df[col].astype(str).fillna('Missing')
                df[col] = encoders[col].fit_transform(df[col])
                feature_cols.append(col)
                numeric_cols_for_scaling.append(col)  # Nach dem Encoden ist es eine Zahl, also ab in den Scaler
            else:
                # Es waren Zahlen im Textformat!
                df[col] = temp_numeric.fillna(0)
                feature_cols.append(col)
                numeric_cols_for_scaling.append(col)

    if len(numeric_cols_for_scaling) > 0:
        scaler = MinMaxScaler()
        df[numeric_cols_for_scaling] = scaler.fit_transform(df[numeric_cols_for_scaling])

    print(f"Dynamische Feature-Erkennung abgeschlossen! Das Modell nutzt nun {len(feature_cols)} Features.")
    return df, scaler, station_encoder, reason_encoder, feature_cols


def create_sequences_multivar(df, feature_cols):
    X, y_when, y_where, y_why = [], [], [], []
    feature_data = df[feature_cols].values
    when_data = df['is_failure'].values
    where_data = df['station_encoded'].values
    why_data = df['reason_encoded'].values

    for i in range(len(df) - p.SEQ_LENGTH):
        X.append(feature_data[i: i + p.SEQ_LENGTH])
        y_when.append(when_data[i + p.SEQ_LENGTH])
        y_where.append(where_data[i + p.SEQ_LENGTH])
        y_why.append(why_data[i + p.SEQ_LENGTH])

    return np.array(X), np.array(y_when), np.array(y_where), np.array(y_why)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
    ax1.set_title('Total Model Loss')
    ax1.legend();
    ax1.grid(True, linestyle='--')

    if 'When_Failure_accuracy' in history.history:
        ax2.plot(history.history['When_Failure_accuracy'], label='Train Acc', color='green')
        ax2.plot(history.history['val_When_Failure_accuracy'], label='Val Acc', color='red')
        ax2.set_title('Accuracy: Failure Prediction')
        ax2.legend();
        ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\trainingsverlauf.png')
   # plt.show()