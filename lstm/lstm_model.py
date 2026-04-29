# main.py
import os

# Unterdrückt die TensorFlow C++ Warnungen (oneDNN, CPU instructions)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import parameters as p
import helping_functions as hf

if __name__ == "__main__":
    # =========================================================================
    # Model Pipeline
    # =========================================================================
    print("1. Loading and preprocessing data...")
    try:
        df, scaler, station_enc, feature_cols, raw_durations, timestamps = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    print("2. Creating LSTM sequences...")
    X, y_when, y_where, y_duration, seq_timestamps = hf.create_sequences_multivar(
        df, feature_cols, raw_durations, timestamps
    )
    num_features = len(feature_cols)

    # Chronological train-test split
    split_idx = int(len(X) * (1 - p.TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]

    y_when_train, y_when_test = y_when[:split_idx], y_when[split_idx:]
    y_where_train, y_where_test = y_where[:split_idx], y_where[split_idx:]

    # y_duration übernimmt jetzt die Rolle der Zielvariable für die Länge
    y_duration_train, y_duration_test = y_duration[:split_idx], y_duration[split_idx:]

    seq_timestamps_test = seq_timestamps[split_idx:]

    print("3. Computing class weights to counteract data imbalance...")
    classes = np.unique(y_when_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    print(f"Calculated Weights -> No Failure (0): {class_weight_dict[0]:.2f}, Failure (1): {class_weight_dict[1]:.2f}")

    sample_weights_when = np.array([class_weight_dict[label] for label in y_when_train])
    sample_weights_where = np.ones(len(y_where_train))
    sample_weights_duration = np.ones(len(y_duration_train))

    sample_weights_dict = {
        'When_Failure': sample_weights_when,
        'Where_Station': sample_weights_where,
        'Duration': sample_weights_duration
    }

    print("4. Building and compiling the Multi-Output LSTM model...")
    model = hf.build_predictive_maintenance_model(
        input_shape=(p.SEQ_LENGTH, num_features),
        num_stations=len(station_enc.classes_)
    )

    print("5. Starting training with sample weights...")
    history = model.fit(
        X_train,
        {'When_Failure': y_when_train, 'Where_Station': y_where_train, 'Duration': y_duration_train},
        validation_data=(
            X_test,
            {'When_Failure': y_when_test, 'Where_Station': y_where_test, 'Duration': y_duration_test}
        ),
        epochs=p.EPOCHS,
        batch_size=p.BATCH_SIZE,
        verbose=1,
        sample_weight=sample_weights_dict
    )

    print("6. Plotting training curves...")
    hf.plot_training_history(history)

    print("\n--- 7a. MODEL EVALUATION (METRICS) ---")
    predictions = model.predict(X_test)
    pred_when_classes = (predictions['When_Failure'] > 0.5).astype(int).flatten()

    # Die vorhergesagte Dauer als Array
    pred_durations = predictions['Duration'].flatten()

    print(classification_report(y_when_test, pred_when_classes, labels=[0, 1], target_names=["No Failure", "Failure"],
                                zero_division=0))

    # --- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT ---
    print("\n--- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT ---")

    # Decoding Station
    if len(station_enc.classes_) > 1:
        pred_where_classes = np.argmax(predictions['Where_Station'], axis=1)
    else:
        pred_where_classes = (predictions['Where_Station'] > 0.5).astype(int).flatten()

    # Auswahl einiger interessanter Beispiele aus den Testdaten
    actual_failures = np.where(y_when_test == 1)[0]
    predicted_failures = np.where(pred_when_classes == 1)[0]
    true_negatives = np.where((y_when_test == 0) & (pred_when_classes == 0))[0]
    false_positives = [idx for idx in predicted_failures if idx not in actual_failures]

    # Wir mischen ein paar Treffer, Fehlalarme und unauffällige Laufzeiten zusammen
    sample_indices = []
    sample_indices.extend(actual_failures[:5])
    sample_indices.extend(false_positives[:3])
    sample_indices.extend(true_negatives[:2])

    # Bereinigen und sortieren für zeitlichen Ablauf
    sample_indices = sorted(list(set(sample_indices)))

    if len(sample_indices) > 0:
        # Kopfzeile neu formatiert für die Dauer
        header_format = "{:<20} | {:<18} | {:<18} | {:<18} | {:<15} | {:<30}"
        print(header_format.format("Zeitraum/Datum", "Modell: Ausfall?", "Realität: Ausfall?", "Vorhersage Dauer",
                                   "Reale Dauer", "Station (Vorhersage -> Real)"))
        print("-" * 132)

        for idx in sample_indices:
            date_val = str(seq_timestamps_test[idx])[:19]

            # Zeiten
            real_dur = y_duration_test[idx]
            pred_dur = pred_durations[idx]

            pred_when_str = "Ja (Ausfall)" if pred_when_classes[idx] == 1 else "Nein (Läuft)"
            real_when_str = "Ja (Ausfall)" if y_when_test[idx] == 1 else "Nein (Läuft)"

            # Stationen Text
            pred_station = station_enc.inverse_transform([pred_where_classes[idx]])[0]
            real_station = station_enc.inverse_transform([y_where_test[idx]])[0]

            station_str = f"{pred_station} -> {real_station}"

            # Formatting der Minuten
            dur_str = f"{real_dur} Min." if real_dur > 0 else "-"
            # Wir zeigen die vorhergesagte Dauer immer an, um zu sehen, was das Modell schätzt (auch wenn es keinen Ausfall vorhersagt)
            pred_dur_str = f"{pred_dur:.1f} Min." if pred_when_classes[idx] == 1 else f"({pred_dur:.1f} Min.)"

            print(header_format.format(date_val, pred_when_str, real_when_str, pred_dur_str, dur_str, station_str[:28]))
    else:
        print("Es gab in den Testdaten keine auswertbaren Vorhersagen/Ausfälle für dieses Sample.")

    # =========================================================================
    # Explainable AI (XAI) Methods
    # =========================================================================
    hf.generate_lime_explanation(model, X_train, X_test, y_when_test, feature_cols, num_features)
    hf.generate_shap_explanation(model, X_train, X_test, y_when_test, feature_cols)
    hf.calculate_permutation_importance(model, X_test, y_when_test, feature_cols)