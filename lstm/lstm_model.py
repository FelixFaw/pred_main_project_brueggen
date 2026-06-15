# main.py
import os

from sklearn.model_selection import train_test_split

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
        df, scaler, feature_cols, timestamps = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    print("2. Creating LSTM sequences and splitting chronologically (Original Context)...")
    X, y_when, seq_timestamps = hf.create_sequences_multivar(
        df, feature_cols, timestamps
    )
    num_features = len(feature_cols)

    # Strikte chronologische Sortierung
    sort_idx = np.argsort(seq_timestamps)
    X = X[sort_idx]
    y_when = y_when[sort_idx]
    seq_timestamps = seq_timestamps[sort_idx]

    # Chronologischer Split (Die letzten 20% für den unberührten Test)
    total_samples = len(X)
    split_index = int(total_samples * (1 - p.TEST_SPLIT))

    # Direktes, lückenloses Slicing ohne physisches Löschen von Daten!
    X_train = X[:split_index]
    y_when_train = y_when[:split_index]

    X_test = X[split_index:]
    y_when_test = y_when[split_index:]
    seq_timestamps_test = seq_timestamps[split_index:]

    print(f"Anzahl Ausfallfenster im lückenlosen TRAIN-Set: {np.sum(y_when_train)} von {len(y_when_train)} Stunden")
    print(f"Anzahl Ausfallfenster im lückenlosen TEST-Set: {np.sum(y_when_test)} von {len(y_when_test)} Stunden")
    # =========================================================================
    # Random Split
    # =========================================================================

    #X_train, X_test, y_when_train, y_when_test, _, seq_timestamps_test = train_test_split(
     #   X, y_when, seq_timestamps,
     #   test_size=p.TEST_SPLIT,
     #   stratify=y_when,  # Zwingt das Testset dazu, echte Ausfälle zu beinhalten
     #   random_state=42
    #)


    # =========================================================================
    # Chronologischer Split (Kein Random Shuffling)
    # =========================================================================
  #  total_samples = len(X)
  #  split_index = int(total_samples * (1 - p.TEST_SPLIT))

    # Trainingsdaten: Die ersten (1 - TEST_SPLIT) % der Daten
 #   X_train = X[:split_index]
 #   y_when_train = y_when[:split_index]

    # Testdaten: Die letzten X % der Daten (Zukunft)
  #  X_test = X[split_index:]
  #  y_when_test = y_when[split_index:]
  #  seq_timestamps_test = seq_timestamps[split_index:]


    print(f"Anzahl Ausfälle im TRAIN-Set: {np.sum(y_when_train)}")
    print(f"Anzahl Ausfälle im TEST-Set: {np.sum(y_when_test)}")

    print("3. Computing class weights to counteract data imbalance...")
    classes = np.unique(y_when_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}

    # Hier überschreiben wir die globalen Parameter für deine Custom Loss Funktion
    p.NEG_WEIGHT = float(class_weight_dict[0])
    p.POS_WEIGHT = float(class_weight_dict[1])

    print(f" -> Gewicht für Normalbetrieb (0): {p.NEG_WEIGHT:.4f}")
    print(f" -> Gewicht für Ausfallklasse (1): {p.POS_WEIGHT:.4f}")

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
        callbacks=[early_stopping]
    )

    print("6. Plotting training curves...")
    hf.plot_training_history(history)

    print("\n--- 7a. MODEL EVALUATION (METRICS WITH OPTIMIZED THRESHOLD) ---")
    predictions = model.predict(X_test)

    # Der Schlüssel zum Erfolg: Wir nutzen dieselbe 30%-Schwelle wie in der Tabelle
    best_threshold = p.best_threshold
    pred_when_classes = (predictions > best_threshold).astype(int).flatten()

    print(f"Evaluierung mit optimierter Vorhersage-Schwelle (Threshold: {best_threshold * 100}%)")
    print(classification_report(y_when_test, pred_when_classes, labels=[0, 1],
                                target_names=["No Failure", "Failure"], zero_division=0))


    # Auswahl einiger interessanter Beispiele aus den Testdaten (nur basierend auf WHEN)
   # actual_failures = np.where(y_when_test == 1)[0]
   # predicted_failures = np.where(pred_when_classes == 1)[0]
   # true_negatives = np.where((y_when_test == 0) & (pred_when_classes == 0))[0]
   # false_positives = [idx for idx in predicted_failures if idx not in actual_failures]

    # Mischen der Beispiele für die Tabelle
   # sample_indices = []
    #sample_indices.extend(actual_failures[:10])
    #sample_indices.extend(false_positives[:6])
    #sample_indices.extend(true_negatives[:4])

    # Bereinigen und sortieren für zeitlichen Ablauf
   # sample_indices = sorted(list(set(sample_indices)))
    # Nimmt einfach die allerersten 20 fortlaufenden Sequenzen aus dem Test-Set

    # Wählt 20 zufällige Indizes aus dem gesamten Test-Set aus und sortiert sie chronologisch
   # sample_indices = sorted(random.sample(range(len(y_when_test)), min(20, len(y_when_test))))

    # =========================================================================
    # --- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT (BEGRENZT AUF 20) ---
    # =========================================================================
    print("\n--- 7b. PRAXISBEISPIELE: VORHERSAGE VS. REALITÄT (STICHPROBE VON 20 STUNDEN) ---")

    # Festlegen der optimierten Entscheidungsschwelle (Threshold)
    best_threshold = p.best_threshold  # XX% Wahrscheinlichkeit reicht als Ausfall-Warnung

    # Indizes für echte Ausfälle und Normalbetrieb trennen
    actual_failures = np.where(y_when_test == 1)[0]
    true_negatives = np.where(y_when_test == 0)[0]

    sample_indices = []
    # 10 zufällige Ausfallstunden ziehen (falls vorhanden)
    if len(actual_failures) > 0:
        sample_indices.extend(random.sample(list(actual_failures), min(6, len(actual_failures))))
    # 10 zufällige normale Betriebsstunden ziehen
    if len(true_negatives) > 0:
        sample_indices.extend(random.sample(list(true_negatives), min(14, len(true_negatives))))

    # Fallback, falls das Testset leer sein sollte
    if len(sample_indices) == 0:
        sample_indices = random.sample(range(len(y_when_test)), min(20, len(y_when_test)))

    # Chronologisch sortieren, um den zeitlichen Ablauf in der Tabelle zu wahren
    sample_indices = sorted(sample_indices)

    if len(sample_indices) > 0:
        header_format = "{:<25} | {:<18} | {:<19} | {:<40}"
        print(header_format.format("Zeitpunkt (Stunde)", "Modell: Ausfall?", "Realität: Ausfall?",
                                   "Wie sicher ist die Vorhersage in %"))
        print("-" * 110)

        for idx in sample_indices:
            # Holt den formatierten String direkt aus dem Array (YYYY-MM-DD HH:MM:SS)
            date_val = str(seq_timestamps_test[idx])

            # Wahrscheinlichkeit für einen Ausfall aus dem Sigmoid-Output holen
            fail_probability = float(predictions[idx][0])

            # Klassifikation basierend auf dem optimierten Threshold von 30%
            if fail_probability > best_threshold:
                pred_when_str = "Ja (Ausfall)"
                confidence = fail_probability * 100
            else:
                pred_when_str = "Nein (Läuft)"
                confidence = (1 - fail_probability) * 100

            real_when_str = "Ja (Ausfall)" if y_when_test[idx] == 1 else "Nein (Läuft)"
            confidence_str = f"{confidence:.1f}%"

            print(header_format.format(date_val, pred_when_str, real_when_str, confidence_str))
    else:
        print("Es gab in den Testdaten keine auswertbaren Vorhersagen.")
    # =========================================================================
    # Explainable AI (XAI) Methods
    # =========================================================================
    hf.generate_lime_explanation(model, X_train, X_test, y_when_test, feature_cols, num_features)
    hf.generate_shap_explanation(model, X_train, X_test, y_when_test, feature_cols)
    hf.calculate_permutation_importance(model, X_test, y_when_test, feature_cols)