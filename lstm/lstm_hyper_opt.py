import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
import numpy as np

# Projekt-Dateien importieren
import parameters as p
import helping_functions_only_faults as hf


# =========================================================================
# Custom Tuner für dynamische Datenübergabe & Class Weights
# =========================================================================
class LSTMTimeSeriesTuner(kt.RandomSearch):
    """
    Ein benutzerdefinierter Tuner, der vor jedem Trainingsdurchlauf
    die Sequenzen (seq_length) und die Klassengewichte dynamisch anpasst.
    """

    def run_trial(self, trial, df, feature_cols, timestamps, *args, **kwargs):
        hp = trial.hyperparameters

        # 1. Sequenzlänge für diesen Trial abfragen und global überschreiben
        current_seq_length = hp.get('seq_length')
        p.SEQ_LENGTH = current_seq_length

        # 2. Daten dynamisch schneiden (Angepasst für das neue Output-Format inkl. y_dur)
        X, y_when, seq_timestamps, y_dur = hf.create_sequences_multivar(df, feature_cols, timestamps)

        # WICHTIG: Strikte chronologische Sortierung
        sort_idx = np.argsort(seq_timestamps)
        X = X[sort_idx]
        y_when = y_when[sort_idx]
        seq_timestamps = seq_timestamps[sort_idx]
        # (y_dur muss hier nicht zwingend sortiert werden, da der Tuner die Tabelle nicht printet,
        # aber wir machen es der Vollständigkeit halber)
        y_dur = y_dur[sort_idx]

        # 3. Chronologischer Split (Die letzten 20% für den unberührten Test)
        split_idx = int(len(X) * (1 - p.TEST_SPLIT))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_when_train, y_when_test = y_when[:split_idx], y_when[split_idx:]

        # 4. Mathematisch ausbalancierte Basis-Gewichte berechnen
        classes = np.unique(y_when_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
        base_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}

        # 5. Tuning-Multiplikator für die Fehlerklasse anwenden
        failure_multiplier = hp.get('failure_multiplier')
        class_weight_dict = {
            0: base_weight_dict[0],
            1: base_weight_dict[1] * failure_multiplier
        }

        # 6. Hyperparameter & Gewichte in die Trainings-kwargs injizieren
        current_batch_size = hp.Choice('batch_size', values=[32, 64, 128])
        kwargs['batch_size'] = current_batch_size
        kwargs['validation_data'] = (X_test, y_when_test)
        kwargs['class_weight'] = class_weight_dict

        return super(LSTMTimeSeriesTuner, self).run_trial(trial, X_train, y_when_train, *args, **kwargs)


# =========================================================================
# Hypermodell-Funktion
# =========================================================================
def build_ultimate_tuning_model(hp):
    # Wird aus den fixen Hyperparametern geholt
    num_features = hp.get('num_features')

    # 'seq_length' wird hier definiert/registriert
    current_seq_length = hp.Int('seq_length', min_value=12, max_value=48, step=12)

    # L2 Regularisierung hinzufügen (Tuning-Option)
    hp_l2_rate = hp.Choice('l2_rate', values=[1e-2, 1e-3, 1e-4])

    inputs = Input(shape=(current_seq_length, num_features), name="Feature_Input")

    # Suchraum für das Modell
    hp_lstm_1 = hp.Int('lstm_1_units', min_value=16, max_value=128, step=16)
    x = LSTM(units=hp_lstm_1, return_sequences=True, kernel_regularizer=l2(hp_l2_rate))(inputs)

    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=hp_dropout_1)(x)

    hp_lstm_2 = hp.Int('lstm_2_units', min_value=16, max_value=64, step=16)
    x = LSTM(units=hp_lstm_2, return_sequences=False, kernel_regularizer=l2(hp_l2_rate))(x)

    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=hp_dropout_2)(x)

    x = BatchNormalization()(x)

    hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    dense_when = Dense(units=hp_dense_units, activation='relu', name='Dense_When', kernel_regularizer=l2(hp_l2_rate))(x)
    dense_when = BatchNormalization()(dense_when)

    outputs = Dense(1, activation='sigmoid', name='When_Failure')(dense_when)

    model = Model(inputs=inputs, outputs=outputs)

    # Tuning des Gewichtungs-Multiplikators
    hp.Float('failure_multiplier', min_value=1.0, max_value=2.5, step=0.5)

    # Suchraum für Optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')]
    )
    return model


# =========================================================================
# Tuning-Pipeline Ausführung
# =========================================================================
if __name__ == "__main__":
    print("1. Loading and preprocessing base dataframe...")
    try:
        # Angepasst für 5 Rückgabewerte (inkl. mean_duration)
        df, scaler, feature_cols, timestamps, mean_duration = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    num_features = len(feature_cols)

    hp = kt.HyperParameters()
    hp.Fixed('num_features', num_features)

    # Instanziierung des Zeitreihen-Tuners
    tuner = LSTMTimeSeriesTuner(
        hypermodel=build_ultimate_tuning_model,
        hyperparameters=hp,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=30,  # Leicht erhöht für den neuen L2-Parameter
        directory=r'C:\Users\louis\PycharmProjects\pred_main_project_brueggen\lstm',
        project_name='lstm_tuning_auc_v2'
    )

    # Early Stopping Callback definieren
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=8,  # Leicht erhöht, da L2 Regularisierung manchmal etwas länger braucht, um sich einzupendeln
        restore_best_weights=True
    )

    print("\n--- Starting Hyperparameter Search (Optimizing for ROC-AUC) ---")
    tuner.search(
        df=df,
        feature_cols=feature_cols,
        timestamps=timestamps,
        epochs=30,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\n--- Ultimate Tuning Results Summary ---")
    tuner.results_summary()

    # Beste Hyperparameter ausgeben
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBeste Parameter (nach höchstem ROC-AUC) gefunden:")
    print(f"- Sequence Length: {best_hps.get('seq_length')}")
    print(f"- LSTM 1 Units: {best_hps.get('lstm_1_units')}")
    print(f"- Dropout 1: {best_hps.get('dropout_1')}")
    print(f"- LSTM 2 Units: {best_hps.get('lstm_2_units')}")
    print(f"- Dropout 2: {best_hps.get('dropout_2')}")
    print(f"- Dense Units: {best_hps.get('dense_units')}")
    print(f"- Batch Size: {best_hps.get('batch_size')}")
    print(f"- Learning Rate: {best_hps.get('learning_rate')}")
    print(f"- L2 Regularization Rate: {best_hps.get('l2_rate')}")
    print(f"- Failure Class Multiplier: {best_hps.get('failure_multiplier')}")