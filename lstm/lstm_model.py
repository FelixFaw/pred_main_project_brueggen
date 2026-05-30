# main.py
import os

# Unterdrückt die TensorFlow C++ Warnungen (oneDNN, CPU instructions)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import random
import parameters as p
import helping_functions as hf

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

if __name__ == "__main__":
    # =========================================================================
    # Model Pipeline
    # =========================================================================
    print("1. Loading and preprocessing data...")
    try:
        df, scaler, feature_cols, timestamps = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    print("2. Creating LSTM sequences...")
    X, y_when, seq_timestamps = hf.create_sequences_multivar(
        df, feature_cols, timestamps
    )
    num_features = len(feature_cols)

    # Chronological train-test split
    split_idx = int(len(X) * (1 - p.TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_when_train, y_when_test = y_when[:split_idx], y_when[split_idx:]
    seq_timestamps_test = seq_timestamps[split_idx:]

    print("3. Computing class weights to counteract data imbalance...")
    classes = np.unique(y_when_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    print(f"Calculated Weights -> No Failure (0): {class_weight_dict[0]:.2f}, Failure (1): {class_weight_dict[1]:.2f}")

    sample_weights_when = np.array([class_weight_dict[label] for label in y_when_train])

    print("4. Building and compiling the Single-Output LSTM model (Only WHEN)...")
    model = hf.build_predictive_maintenance_model(
        input_shape=(p.SEQ_LENGTH, num_features)
    )

    print("5. Starting training with sample weights...")

    # Early Stopping Callback definieren
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True  # Setzt das Modell am Ende auf den besten Stand zurück
    )

    history = model.fit(
        X_train,
        y_when_train,
        validation_data=(X_test, y_when_test),
        epochs=p.EPOCHS,
        batch_size=p.BATCH_SIZE,
        verbose=1,
        sample_weight=sample_weights_when,
        callbacks = [early_stopping]
    )

    print("6. Plotting training curves...")
    hf.plot_training_history(history)

    print("\n--- 7a. MODEL EVALUATION (METRICS) ---")
    predictions = model.predict(X_test)
    pred_when_classes = (predictions > 0.5).astype(int).flatten()

    print(classification_report(y_when_test, pred_when_classes, labels=[0, 1], target_names=["No Failure", "Failure"],
                                zero_division=0))

    # --- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT ---
    print("\n--- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT ---")

    # Auswahl einiger interessanter Beispiele aus den Testdaten (nur basierend auf WHEN)
    actual_failures = np.where(y_when_test == 1)[0]
    predicted_failures = np.where(pred_when_classes == 1)[0]
    true_negatives = np.where((y_when_test == 0) & (pred_when_classes == 0))[0]
    false_positives = [idx for idx in predicted_failures if idx not in actual_failures]

    # Mischen der Beispiele für die Tabelle
    sample_indices = []
    sample_indices.extend(actual_failures[:5])
    sample_indices.extend(false_positives[:3])
    sample_indices.extend(true_negatives[:2])

    # Bereinigen und sortieren für zeitlichen Ablauf
    sample_indices = sorted(list(set(sample_indices)))

    if len(sample_indices) > 0:
        # Kopfzeile exakt nach Vorgabe aufgebaut, inklusive Prozent-Sicherheit (Modell-Konfidenz)
        header_format = "{:<20} | {:<18} | {:<19} | {:<40}"
        print(header_format.format("Zeitraum/Datum", "Modell: Ausfall?", "Realität: Ausfall?",
                                   "Wie sicher ist die Vorhersage in %"))
        print("-" * 105)

        for idx in sample_indices:
            date_val = str(seq_timestamps_test[idx])[:19]

            # Wahrscheinlichkeit für einen Ausfall aus dem Sigmoid-Output holen
            fail_probability = float(predictions[idx][0])

            if pred_when_classes[idx] == 1:
                pred_when_str = "Ja (Ausfall)"
                confidence = fail_probability * 100
            else:
                pred_when_str = "Nein (Läuft)"
                confidence = (1 - fail_probability) * 100

            real_when_str = "Ja (Ausfall)" if y_when_test[idx] == 1 else "Nein (Läuft)"
            confidence_str = f"{confidence:.1f}%"

            print(header_format.format(date_val, pred_when_str, real_when_str, confidence_str))
    else:
        print("Es gab in den Testdaten keine auswertbaren Vorhersagen/Ausfälle für dieses Sample.")

    # =========================================================================
    # Explainable AI (XAI) Methods
    # =========================================================================
    hf.generate_lime_explanation(model, X_train, X_test, y_when_test, feature_cols, num_features)
    hf.generate_shap_explanation(model, X_train, X_test, y_when_test, feature_cols)
    hf.calculate_permutation_importance(model, X_test, y_when_test, feature_cols)