# helping_functions.py
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Importe für Modellbau und XAI
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from lime import lime_tabular
import shap

import parameters as p


def load_and_preprocess_data_all_features():
    print("Lade Prozessdaten...")

    file_path = p.FILE_PATH_FAULTS

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    else:
        df = pd.read_excel(file_path)

    # ---------------------------------------------------------
    # 1. Zielvariablen (Targets) extrahieren und BEREINIGEN
    # ---------------------------------------------------------
    col_failure_duration = 'Dauer Anlagen-Ausfall'
    col_station = 'Station/ OP'

    print("Bereite Zielvariable (Target: WHEN) vor...")

    # Dauer-Spalte bereinigen, um das 'WHEN'-Label (is_failure) sauber zu erzeugen
    if col_failure_duration not in df.columns:
        df[col_failure_duration] = 0
    else:
        df[col_failure_duration] = pd.to_numeric(df[col_failure_duration], errors='coerce').fillna(0)

    # Das einzige verbleibende Target: Hat ein Ausfall stattgefunden?
    is_failure = (df[col_failure_duration] > 0).astype(int)

    if p.DT_FAULTS in df.columns:
        timestamps = df[p.DT_FAULTS].astype(str).values
    elif 'Istenddat.Durchf.' in df.columns:
        timestamps = df['Istenddat.Durchf.'].astype(str).values
    else:
        timestamps = np.array([f"Schritt {i}" for i in range(len(df))])

    # ---------------------------------------------------------
    # 2. Relevante Features definieren
    # ---------------------------------------------------------
    categorical_features = [
        'Schicht', 'Materialnummer', 'Auftrag', 'Material-Text',
        'Arbeitsplatz', 'Systemstatus'
    ]

    numeric_features = [
        'Wochentag', 'Anzahl MA', 'Menge N.i. O.', 'Menge i. O. L4',
        'Menge i. O. L5', 'Menge Gesamt (Stück)', 'Dauer Org-Mangel',
        'Dauer Anlagen-Ausfall intern', 'Dauer Logistik- Defizite',
        'Sollzeit/ Stück (Min)', 'Takt Gesamt', 'Zeit_von_min', 'Zeit_bis_min',
        'Vorgangsmenge (MEINH)', 'Rückgem. Gutmenge (MEINH)',
        'Vorgabewert 2 (VGE02)', 'Vorgabewert 3 (VGE03)'
    ]

    # Data Leakage verhindern (Sowohl Dauer als auch Station komplett ausschließen!)
    numeric_features = [f for f in numeric_features if f not in [col_failure_duration, col_station]]
    categorical_features = [f for f in categorical_features if f not in [col_failure_duration, col_station]]

    num_cols_present = [col for col in numeric_features if col in df.columns]
    cat_cols_present = [col for col in categorical_features if col in df.columns]

    # ---------------------------------------------------------
    # 3. Vorverarbeitung
    # ---------------------------------------------------------
    print("Fülle leere Werte in den Features...")
    for col in num_cols_present:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for cat in cat_cols_present:
        df[cat] = df[cat].fillna('unbekannt').astype(str)

    print("Wandle Kategorien in numerische Label um...")
    encoded_cat_cols = []
    for cat in cat_cols_present:
        le_feature = LabelEncoder()
        col_name = f'{cat}_encoded'
        df[col_name] = le_feature.fit_transform(df[cat])
        encoded_cat_cols.append(col_name)

    all_features = num_cols_present + encoded_cat_cols

    # ---------------------------------------------------------
    # 4. MinMaxScaler anwenden
    # ---------------------------------------------------------
    print("Skaliere Features...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[all_features])

    df_final = pd.DataFrame(scaled_data, columns=all_features)
    df_final['is_failure'] = is_failure.values

    target_cols = ['is_failure']
    feature_cols = [col for col in df_final.columns if col not in target_cols]

    return df_final, scaler, feature_cols, timestamps


def create_sequences_multivar(df, feature_cols, timestamps):
    X, y_when = [], []
    seq_timestamps = []

    feature_data = df[feature_cols].values
    when_data = df['is_failure'].values

    for i in range(len(df) - p.SEQ_LENGTH):
        target_idx = i + p.SEQ_LENGTH

        X.append(feature_data[i: target_idx])
        y_when.append(when_data[target_idx])
        seq_timestamps.append(timestamps[target_idx])

    return np.array(X), np.array(y_when), np.array(seq_timestamps)


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
    plt.savefig(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\trainingsverlauf.png')
    plt.close()


def weighted_binary_crossentropy():
    """
    Benutzerdefinierte Loss-Funktion für unbalancierte binäre Daten.
    pos_weight: Gewicht für die Fehlerklasse (1 / Failure)
    neg_weight: Gewicht für den Normalbetrieb (0 / No Failure)
    """

    def loss(y_true, y_pred):
        # Datentypen anpassen und Werte clippen, um log(0) - Fehler zu vermeiden
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Mathematische Berechnung der gewichteten Kreuzentropie
        bin_los_1 = y_true * K.log(y_pred) * p.POS_WEIGHT
        bin_los_0 = (1.0 - y_true) * K.log(1.0 - y_pred) * p.NEG_WEIGHT

        return -K.mean(bin_los_1 + bin_los_0, axis=-1)

    return loss

def build_predictive_maintenance_model(input_shape):
    inputs = Input(shape=input_shape, name="Feature_Input")

    x = LSTM(16, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.4)(x)

    x = BatchNormalization()(x)


    # Lineare Single-Output-Architektur statt Verzweigung
    dense_when = Dense(48, activation='relu', name='Dense_When')(x)

    dense_when = BatchNormalization()(dense_when)

    outputs = Dense(1, activation='sigmoid', name='When_Failure')(dense_when)

    model = Model(inputs=inputs, outputs=outputs)

    # Einfacher Kompiliervorgang, da es nur noch einen Verlust (Loss) gibt
    model.compile(
        optimizer=Adam(learning_rate=p.LEARNING_RATE),
        loss=weighted_binary_crossentropy(),
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
            num_features=14,
            labels=(1,)
        )
        exp.save_to_file(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\lime_explanation_failure.html')
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

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_2d, test_samples_2d, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\shap_summary_plot.png')
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
        np.random.shuffle(X_test_shuffled[:, :, i])

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
    plt.savefig(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\permutation_importance.png')
    plt.close()