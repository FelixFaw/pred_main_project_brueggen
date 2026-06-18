import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve,roc_auc_score, auc as sklearn_auc
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import parameters as p


def load_and_preprocess_data_all_features():
    print("Lade Daten und wende strikte Feature-Trennung (Predictive Maintenance) an...")
    df_faults = pd.read_csv(p.FILE_PATH_FAULTS, sep=';', encoding='utf-8')

    df_faults = df_faults.dropna(subset=[p.DT_FAULTS, 'Zeit von'])
    df_faults['merge_date'] = pd.to_datetime(
        df_faults[p.DT_FAULTS].astype(str).str.strip() + ' ' + df_faults['Zeit von'].astype(str).str.strip(),
        format='mixed', errors='coerce'
    ).dt.floor('h')
    df_faults = df_faults.dropna(subset=['merge_date'])

    num_cols = [c for c in p.NUMERIC_FEATURES if c in df_faults.columns]
    cat_cols = [c for c in p.CATEGORICAL_FEATURES if c in df_faults.columns]
    target_cols = [c for c in p.TARGETS if c in df_faults.columns]

    for col in num_cols + target_cols:
        df_faults[col] = pd.to_numeric(df_faults[col], errors='coerce').fillna(0)

    if cat_cols:
        df_faults = pd.get_dummies(df_faults, columns=cat_cols, drop_first=True, dtype=int)
        dummy_cols = [c for c in df_faults.columns if any(c.startswith(orig_cat) for orig_cat in cat_cols)]
    else:
        dummy_cols = []

    agg_dict = {}
    for col in num_cols + dummy_cols + target_cols:
        if any(keyword in col for keyword in ['Dauer', 'Menge', 'Anzahl', 'Schicht_', 'Station/', 'Bemerkung_']):
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'mean'

    df_hourly = df_faults.groupby('merge_date').agg(agg_dict).reset_index()
    df_hourly = df_hourly.sort_values('merge_date').reset_index(drop=True)

    if 'Anzahl Störfälle Zeitfenster' in df_hourly.columns:
        base_trend_col = 'Anzahl Störfälle Zeitfenster'
    else:
        df_hourly['Anzahl_Ausfaelle'] = 1
        base_trend_col = 'Anzahl_Ausfaelle'

    df_hourly['Ausfaelle_Schwankung_3h'] = df_hourly[base_trend_col].rolling(window=3, min_periods=1).std().fillna(0)
    df_hourly['Ausfaelle_Schwankung_6h'] = df_hourly[base_trend_col].rolling(window=6, min_periods=1).std().fillna(0)

    df = df_hourly.copy()
    is_failure = (df[p.TARGET_DURATION] > 0).astype(int)

    # Zombie-Features entfernen, da sie keinen Mehrwert bringen
    df.drop(columns=['Ausfaelle_Schwankung_6h', 'Ausfaelle_Schwankung_3h', 'Anzahl_Ausfaelle',
                     'Anzahl Störfälle Zeitfenster'], errors='ignore', inplace=True)

    timestamps = df['merge_date'].dt.strftime('%Y-%m-%d %H:%M:%S').values

    feature_cols = [c for c in df.columns if c not in ['merge_date'] + p.TARGETS]
    print(f"Stündliche Daten Matrix Form (X-Features): {len(feature_cols)} Spalten")

    train_size = int(len(df) * (1 - p.TEST_SPLIT))
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[:train_size][feature_cols])
    scaled_data = scaler.transform(df[feature_cols])

    df_final = pd.DataFrame(scaled_data, columns=feature_cols)
    df_final['is_failure'] = is_failure.values

    # NEU: Reale Ausfalldauer für das spätere Testset speichern
    df_final['actual_duration'] = df[p.TARGET_DURATION].values

    # NEU: Historischen Durchschnitt für die rein informative Schätzung berechnen
    mean_duration = df[df[p.TARGET_DURATION] > 0][p.TARGET_DURATION].mean()

    return df_final, scaler, feature_cols, timestamps, mean_duration

def create_sequences_multivar(df, feature_cols, timestamps):
    X, y, seq_t, y_dur = [], [], [], []  # y_dur NEU
    feature_data = df[feature_cols].values
    target_data = df['is_failure'].values
    duration_data = df['actual_duration'].values  # NEU
    PREDICTION_WINDOW = 2

    for i in range(len(df) - p.SEQ_LENGTH - PREDICTION_WINDOW + 1):
        X.append(feature_data[i: i + p.SEQ_LENGTH])
        window_targets = target_data[i + p.SEQ_LENGTH: i + p.SEQ_LENGTH + PREDICTION_WINDOW]
        window_durs = duration_data[i + p.SEQ_LENGTH: i + p.SEQ_LENGTH + PREDICTION_WINDOW]  # NEU

        if np.any(window_targets == 1):
            y.append(1)
        else:
            y.append(0)

        y_dur.append(np.sum(window_durs))  # Echte Dauer aufsummieren
        seq_t.append(timestamps[i + p.SEQ_LENGTH - 1])

    return np.array(X), np.array(y), np.array(seq_t), np.array(y_dur)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
    ax1.set_title('Model Loss (Binary Crossentropy)')
    ax1.legend()
    ax1.grid(True, linestyle='--')

    # Optimierter Fokus auf den Verlauf der ROC-AUC Kurve im Training
    if 'auc' in history.history:
        ax2.plot(history.history['auc'], label='Train ROC-AUC', color='green')
        ax2.plot(history.history['val_auc'], label='Val ROC-AUC', color='red')
        ax2.set_title('Model ROC-AUC History')
        ax2.legend()
        ax2.grid(True, linestyle='--')
    elif 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Train Acc', color='green')
        ax2.plot(history.history['val_accuracy'], label='Val Acc', color='red')
        ax2.set_title('Accuracy History')
        ax2.legend()
        ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\trainingshistory.png')
    plt.close()

def build_predictive_maintenance_model(input_shape):
    inputs = Input(shape=input_shape, name="Feature_Input")

    x = LSTM(p.lstm_1_units, return_sequences=True, kernel_regularizer=l2(p.L2_RATE))(inputs)
    x = Dropout(p.dropout_1)(x)

    x = LSTM(p.lstm_2_units, return_sequences=False, kernel_regularizer=l2(p.L2_RATE))(x)
    x = Dropout(p.dropout_2)(x)

    x = BatchNormalization()(x)

    dense_when = Dense(p.dense_units, activation='relu', kernel_regularizer=l2(p.L2_RATE))(x)
    dense_when = BatchNormalization()(dense_when)

    outputs = Dense(1, activation='sigmoid', name='When_Failure')(dense_when)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=p.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')]
    )
    return model

def optimize_threshold_and_plot_roc(y_true, y_probs):
    """
    Berechnet die ROC-Kurve, bestimmt den mathematisch besten Schwellenwert
    mittels Youden-Index und speichert den Plot ab.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = sklearn_auc(fpr, tpr)

    # Youden-Index: J = True Positive Rate - False Positive Rate
    # Wir suchen das Maximum, um den besten Kompromiss aus Sensitivität und Spezifität zu finden
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # ROC-Kurve plotten
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', s=100,
                label=f'Optimaler Threshold = {best_threshold:.3f} (Youden)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Falsch-Alarm-Rate)')
    plt.ylabel('True Positive Rate (Trefferquote)')
    plt.title('Receiver Operating Characteristic (ROC) Kurve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\roc_curve.png')
    plt.close()

    return best_threshold, roc_auc


def calculate_permutation_importance(model, X_test, y_when_test, feature_cols):
    print("\n--- 10. Starting Permutation Feature Importance (AUC-basiert) ---")

    # 1. Baseline Wahrscheinlichkeiten (nicht Klassen!) und Baseline AUC berechnen
    baseline_probs = model.predict(X_test, verbose=0).flatten()
    baseline_auc = roc_auc_score(y_when_test, baseline_probs)

    # 2. Features gruppieren (Zusammenfassen von One-Hot-Encodings)
    feature_groups = {}
    for i, col in enumerate(feature_cols):
        group_assigned = False
        for cat in p.CATEGORICAL_FEATURES:
            if col.startswith(cat + '_'):
                if cat not in feature_groups:
                    feature_groups[cat] = []
                feature_groups[cat].append(i)
                group_assigned = True
                break
        if not group_assigned:
            feature_groups[col] = [i]

    feature_importances = {}
    # 3. Permutation auf Gruppenbasis
    for group_name, indices in feature_groups.items():
        X_test_shuffled = X_test.copy()
        shuffled_idx = np.random.permutation(len(X_test_shuffled))

        for i in indices:
            X_test_shuffled[:, :, i] = X_test_shuffled[shuffled_idx, :, i]

        # 4. AUC mit gemischten Daten berechnen
        shuffled_probs = model.predict(X_test_shuffled, verbose=0).flatten()
        shuffled_auc = roc_auc_score(y_when_test, shuffled_probs)

        # 5. Differenz ist jetzt der AUC-Drop (wie viel schlechter wird das Modell?)
        feature_importances[group_name] = baseline_auc - shuffled_auc

    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=False)
    features_sorted = [x[0] for x in sorted_importances]
    importances_sorted = [x[1] for x in sorted_importances]

    plot_height = max(5, len(features_sorted) * 0.4)
    plt.figure(figsize=(8, plot_height))
    plt.barh(features_sorted, importances_sorted, color='skyblue')
    plt.xlabel("Abfall im ROC-AUC (Importance)")  # Label angepasst!
    plt.title("Permutation Feature Importance (Gruppiert, AUC-Drop)")
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(
        r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm\lstm_output\permutation_importance.png')
    plt.close()