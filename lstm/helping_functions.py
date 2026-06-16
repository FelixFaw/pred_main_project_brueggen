# helping_functions.py

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from lime import lime_tabular
import shap

import parameters as p


def load_and_preprocess_data_all_features():
    print("Lade Daten und generiere hoch-informative Trend-Features...")

    df_process = pd.read_csv(p.FILE_PATH_PROCESS, sep=';', encoding='utf-8')
    df_faults = pd.read_csv(p.FILE_PATH_FAULTS, sep=',', encoding='utf-8')

    # --- 1. SPALTEN-AUSWAHL FÜR PROZESSDATEN ---
    # Priorisiere die neuen Spalten 'Ist-Ende Vorg.' und 'Istendzt.Durchf.'
    if 'Ist-Ende Vorg.' in df_process.columns:
        col_proc_date = 'Ist-Ende Vorg.'
    else:
        col_proc_date = 'Ende Durchf.(Dat.)'

    if 'Istendzt.Durchf.' in df_process.columns:
        col_proc_time = 'Istendzt.Durchf.'
    else:
        col_proc_time = 'Ende Durchf.(Zeit)'

    # Zeilen löschen, bei denen Datum oder Zeit fehlen
    df_process = df_process.dropna(subset=[col_proc_date, col_proc_time])

    # Datum und Zeit zusammenführen und auf volle Stunde runden
    df_process['merge_date'] = pd.to_datetime(
        df_process[col_proc_date].astype(str).str.strip() + ' ' + df_process[col_proc_time].astype(str).str.strip(),
        format='mixed', errors='coerce'
    ).dt.floor('h')
    df_process = df_process.dropna(subset=['merge_date'])

    # --- 2. ZEITSTEMPEL RUNDEN (AUSFALLDATEN) ---
    df_faults = df_faults.dropna(subset=['Datum', 'Zeit von'])
    df_faults['merge_date'] = pd.to_datetime(
        df_faults['Datum'].astype(str).str.strip() + ' ' + df_faults['Zeit von'].astype(str).str.strip(),
        format='mixed', errors='coerce'
    ).dt.floor('h')
    df_faults = df_faults.dropna(subset=['merge_date'])

    # --- 3. BEREINIGUNG DER MENGEN ---
    df_process['Gutmenge_bereinigt'] = df_process['Rückgem. Gutmenge (GMEIN)'].fillna(
        df_process['Rückgem. Gutmenge (MEINH)'])
    df_process['Gutmenge_bereinigt'] = pd.to_numeric(df_process['Gutmenge_bereinigt'], errors='coerce').fillna(0)

    # 4. Stündliche Basis-Features aggregieren
    df_hourly = df_process.groupby('merge_date').agg(
        Durchsatz_Pro_Stunde=('Gutmenge_bereinigt', 'sum'),
        Anzahl_Prozessschritte=('Vorgang', 'count'),
        Materialwechsel_Anzahl=('Materialnummer', 'nunique'),
        Statuswechsel_Anzahl=('Systemstatus', 'nunique')
    ).reset_index()

    df_hourly = df_hourly.sort_values('merge_date').reset_index(drop=True)

    # --- 5. TREND-FEATURES GENERIEREN ---
    df_hourly['Durchsatz_Schwankung_3h'] = df_hourly['Durchsatz_Pro_Stunde'].rolling(window=3,
                                                                                     min_periods=1).std().fillna(0)
    df_hourly['Durchsatz_Schwankung_6h'] = df_hourly['Durchsatz_Pro_Stunde'].rolling(window=6,
                                                                                     min_periods=1).std().fillna(0)

    df_hourly['Prozess_Schwankung_3h'] = df_hourly['Anzahl_Prozessschritte'].rolling(window=3,
                                                                                     min_periods=1).std().fillna(0)
    df_hourly['Prozess_Schwankung_6h'] = df_hourly['Anzahl_Prozessschritte'].rolling(window=6,
                                                                                     min_periods=1).std().fillna(0)

    # Ausfalldaten aggregieren
    df_faults_hourly = df_faults.groupby('merge_date').agg(
        Dauer_Anlagen_Ausfall=('Dauer Anlagen-Ausfall', 'sum')
    ).reset_index()

    # --- 6. DATENTYP-GEWÄHRLEISTUNG VOR DEM MERGE ---
    # Erzwinge bei beiden DataFrames den exakt gleichen Datetime-Datentyp
    df_hourly['merge_date'] = pd.to_datetime(df_hourly['merge_date'])
    df_faults_hourly['merge_date'] = pd.to_datetime(df_faults_hourly['merge_date'])

    # Verheiraten via Left Join
    df = pd.merge(df_hourly, df_faults_hourly, on='merge_date', how='left')
    df = df.sort_values('merge_date').reset_index(drop=True)

    df['Dauer_Anlagen_Ausfall'] = df['Dauer_Anlagen_Ausfall'].fillna(0)
    is_failure = (df['Dauer_Anlagen_Ausfall'] > 0).astype(int)

    # Wichtige Kontroll-Ausgaben im Terminal
    print(f"-> Anzahl gematchter Ausfallstunden nach Merge: {is_failure.sum()}")
    print(f"Stündliche Daten Matrix Form mit Trend-Features: {df.shape}")

    timestamps = df['merge_date'].dt.strftime('%Y-%m-%d %H:%M:%S').values

    # Features isolieren
    feature_cols = [c for c in df.columns if c not in ['merge_date', 'Dauer_Anlagen_Ausfall']]

    # Skalierung
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    df_final = pd.DataFrame(scaled_data, columns=feature_cols)
    df_final['is_failure'] = is_failure.values

    return df_final, scaler, feature_cols, timestamps


def create_sequences_multivar(df, feature_cols, timestamps):
    X, y, seq_t = [], [], []
    feature_data = df[feature_cols].values
    target_data = df['is_failure'].values

    PREDICTION_WINDOW = 2

    for i in range(len(df) - p.SEQ_LENGTH - PREDICTION_WINDOW + 1):
        X.append(feature_data[i: i + p.SEQ_LENGTH])

        window_targets = target_data[i + p.SEQ_LENGTH: i + p.SEQ_LENGTH + PREDICTION_WINDOW]
        if np.any(window_targets == 1):
            y.append(1)
        else:
            y.append(0)

        seq_t.append(timestamps[i + p.SEQ_LENGTH - 1])

    return np.array(X), np.array(y), np.array(seq_t)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
    ax1.set_title('Model Loss (WHEN Only)')
    ax1.legend()
    ax1.grid(True, linestyle='--')

    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Train Acc', color='green')
        ax2.plot(history.history['val_accuracy'], label='Val Acc', color='red')
        ax2.set_title('Accuracy: Failure Prediction')
        ax2.legend()
        ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\trainingshistory.png')
    plt.close()

def build_predictive_maintenance_model(input_shape):
    inputs = Input(shape=input_shape, name="Feature_Input")

    # L2-Regularisierung (0.001) zwingt das Modell zu weichen, stabilen Entscheidungsgrenzen
    x = LSTM(p.lstm_1_units, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
    x = Dropout(p.dropout_1)(x)

    x = LSTM(p.lstm_2_units, return_sequences=False, kernel_regularizer=l2(0.001))(x)
    x = Dropout(p.dropout_2)(x)

    x = BatchNormalization()(x)

    dense_when = Dense(p.dense_units, activation='relu', kernel_regularizer=l2(0.001))(x)
    dense_when = BatchNormalization()(dense_when)

    outputs = Dense(1, activation='sigmoid', name='When_Failure')(dense_when)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=p.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def generate_lime_explanation(model, X_train, X_test, y_when_test, feature_cols, num_features):
    print("\n--- 8. Starting LIME (Explainable AI) Feature Analysis ---")
    lime_feature_names = []
    for timestep in range(p.SEQ_LENGTH):
        for fname in feature_cols:
            lime_feature_names.append(f"{fname} (t-{p.SEQ_LENGTH - 1 - timestep})")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=lime_feature_names,
        class_names=["No Failure", "Failure"],
        mode='classification',
        discretize_continuous=True
    )

    failure_indices = np.where(y_when_test == 1)[0]
    if len(failure_indices) > 0:
        idx_to_explain = failure_indices[0]
        instance_3d = X_test[idx_to_explain]

        def lime_predict_wrapper(flattened_data):
            reshaped_3d = flattened_data.reshape(flattened_data.shape[0], p.SEQ_LENGTH, num_features)
            fail_probs = model.predict(reshaped_3d, verbose=0)
            return np.hstack([1 - fail_probs, fail_probs])

        exp = explainer.explain_instance(
            data_row=instance_3d.reshape(-1),
            predict_fn=lime_predict_wrapper,
            num_features=p.num_features,
            labels=(1,)
        )
        exp.save_to_file(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\lime_explanation_failure.html')
    else:
        print("Note: No machine failures were found in the test dataset for LIME.")


def generate_shap_explanation(model, X_train, X_test, y_when_test, feature_cols):
    print("\n--- 9. Starting SHAP (Shapley Additive exPlanations) ---")

    # Da das Modell nun direkt "When_Failure" ausgibt, greifen wir direkt auf das Haupt-Output zu
    shap_model = model

    background_indices = np.random.choice(X_train.shape[0], min(100, len(X_train)), replace=False)
    background_data = X_train[background_indices]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")
        warnings.filterwarnings("ignore", category=FutureWarning)

        explainer_shap = shap.GradientExplainer(shap_model, background_data)
        failure_indices = np.where(y_when_test == 1)[0]

        if len(failure_indices) > 0:
            test_samples_to_explain = X_test[failure_indices[:20]]
            shap_values = explainer_shap.shap_values(test_samples_to_explain)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values_2d = np.sum(shap_values, axis=1)
            test_samples_2d = np.mean(test_samples_to_explain, axis=1)

            plt.figure(figsize=(1, 3))
            shap.summary_plot(shap_values_2d, test_samples_2d, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\shap_summary_plot.png')
            plt.close()
        else:
            print("Note: No machine failures were found in the test dataset for SHAP.")


def calculate_permutation_importance(model, X_test, y_when_test, feature_cols):
    print("\n--- 10. Starting Permutation Feature Importance ---")
    baseline_preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    baseline_acc = np.mean(baseline_preds == y_when_test)

    feature_importances = {}
    for i, feature_name in enumerate(feature_cols):
        X_test_shuffled = X_test.copy()
        # Für das Feature 'i' die Zuordnung über die Samples hinweg vertauschen
        shuffled_idx = np.random.permutation(len(X_test))
        X_test_shuffled[:, :, i] = X_test_shuffled[shuffled_idx, :, i]

        shuffled_preds = (model.predict(X_test_shuffled, verbose=0) > 0.5).astype(int).flatten()
        shuffled_acc = np.mean(shuffled_preds == y_when_test)
        feature_importances[feature_name] = baseline_acc - shuffled_acc

    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=False)
    features_sorted = [x[0] for x in sorted_importances]
    importances_sorted = [x[1] for x in sorted_importances]

    plt.figure(figsize=(10, 8))
    plt.barh(features_sorted, importances_sorted, color='skyblue')
    plt.xlabel("Abfall in der Genauigkeit (Accuracy Drop)")
    plt.title("Permutation Feature Importance (Global)")
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\permutation_importance.png')
    plt.close()