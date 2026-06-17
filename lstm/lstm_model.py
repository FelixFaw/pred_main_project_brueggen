# main.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import parameters as p
import helping_functions_only_faults as hf

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

if __name__ == "__main__":
    # =========================================================================
    # Model Pipeline
    # =========================================================================
    print("1. Loading and preprocessing data...")
    try:
        df, scaler, feature_cols, timestamps, mean_duration = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    print("2. Creating LSTM sequences and splitting chronologically (Original Context)...")
    X, y_when, seq_timestamps, y_dur = hf.create_sequences_multivar(
        df, feature_cols, timestamps
    )
    num_features = len(feature_cols)

    # Strikte chronologische Sortierung
    sort_idx = np.argsort(seq_timestamps)
    X = X[sort_idx]
    y_when = y_when[sort_idx]
    seq_timestamps = seq_timestamps[sort_idx]
    y_dur = y_dur[sort_idx]  # Echte Dauer sortieren

    # Chronologischer Split
    total_samples = len(X)
    split_index = int(total_samples * (1 - p.TEST_SPLIT))

    X_train = X[:split_index]
    y_when_train = y_when[:split_index]

    X_test = X[split_index:]
    y_when_test = y_when[split_index:]
    seq_timestamps_test = seq_timestamps[split_index:]
    y_dur_test = y_dur[split_index:]  # Dauer fürs Testset trennen

    print("3. Computing class weights to counteract data imbalance...")
    classes = np.unique(y_when_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1] * p.Failure_Class_Multiplier}

    print("4. Building and compiling the Single-Output LSTM model (Optimized for ROC-AUC)...")
    model = hf.build_predictive_maintenance_model(
        input_shape=(p.SEQ_LENGTH, num_features)
    )

    print("5. Starting training with sample weights & Early Stopping on Validation ROC-AUC...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=15,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_when_train,
        validation_data=(X_test, y_when_test),
        epochs=p.EPOCHS,
        batch_size=p.BATCH_SIZE,
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    print("6. Plotting training curves...")
    hf.plot_training_history(history)

    print("\n--- 7a. MODEL EVALUATION (THRESHOLD EVALUATION) ---")
    predictions = model.predict(X_test).flatten()

    # Berechne die ROC-Kurve und den automatischen Optimalwert im Hintergrund
    auto_threshold, roc_auc_score = hf.optimize_threshold_and_plot_roc(y_when_test, predictions)
    print(f"\n>>> Maximal erzielter ROC-AUC Score im Testset: {roc_auc_score:.4f} <<<")
    print(f"-> Mathematisch bester Schwellenwert (Youden-Index): {auto_threshold * 100:.2f}%")

    # Logik zur flexiblen Threshold-Auswahl
    if p.MANUAL_THRESHOLD is not None:
        chosen_threshold = p.MANUAL_THRESHOLD
        print(f"\n[INFO] NUTZE MANUELLEN SCHWELLENWERT: {chosen_threshold * 100:.1f}%")
    else:
        chosen_threshold = auto_threshold
        print(f"\n[INFO] NUTZE AUTOMATISCH OPTIMIERTEN SCHWELLENWERT: {chosen_threshold * 100:.2f}%")

    pred_when_classes = (predictions > chosen_threshold).astype(int)

    print(classification_report(y_when_test, pred_when_classes, labels=[0, 1],
                                target_names=["No Failure", "Failure"], zero_division=0))

    # =========================================================================
    # --- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT (BEGRENZT AUF 20) ---
    # =========================================================================
    print("\n--- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT (STICHPROBE VON 20 STUNDEN) ---")

    actual_failures = np.where(y_when_test == 1)[0]
    true_negatives = np.where(y_when_test == 0)[0]

    sample_indices = []
    if len(actual_failures) > 0:
        sample_indices.extend(random.sample(list(actual_failures), min(6, len(actual_failures))))
    if len(true_negatives) > 0:
        sample_indices.extend(random.sample(list(true_negatives), min(14, len(true_negatives))))

    if len(sample_indices) == 0:
        sample_indices = random.sample(range(len(y_when_test)), min(20, len(y_when_test)))

    sample_indices = sorted(sample_indices)

    if len(sample_indices) > 0:
        # Einmalige Info vor der Tabelle
        print(f"\n[INFO] Die historische Durchschnittsdauer eines Ausfalls beträgt ca. {mean_duration:.0f} Minuten.")
        print("Dieser Wert dient als statistischer Richtwert und wird nicht dynamisch vorhergesagt.\n")

        # Verschlankter Header
        header_format = "{:<20} | {:<16} | {:<16} | {:<15} | {:<15}"
        print(header_format.format("Zeitpunkt", "Modell: Ausfall?", "Real: Ausfall?", "Real: Dauer", "Sicherheit"))
        print("-" * 90)

        for idx in sample_indices:
            date_val = str(seq_timestamps_test[idx])
            fail_probability = float(predictions[idx])
            actual_dur = y_dur_test[idx]

            # Klassifikation ohne angehängte Schätzung
            if fail_probability > chosen_threshold:
                pred_when_str = "Ja (Ausfall)"
                confidence = fail_probability * 100
            else:
                pred_when_str = "Nein (Läuft)"
                confidence = (1 - fail_probability) * 100

            real_when_str = "Ja (Ausfall)" if y_when_test[idx] == 1 else "Nein (Läuft)"
            # Zeigt die echten Minuten, wenn es in der Realität einen Ausfall gab
            actual_dur_str = f"{actual_dur:.0f} min" if actual_dur > 0 else "-"
            conf_str = f"{confidence:.1f}%"

            print(header_format.format(date_val, pred_when_str, real_when_str, actual_dur_str, conf_str))
    else:
        print("Es gab in den Testdaten keine auswertbaren Vorhersagen.")
    # =========================================================================
    # Explainable AI (XAI) Methods
    # =========================================================================
    hf.calculate_permutation_importance(model, X_test, y_when_test, feature_cols)