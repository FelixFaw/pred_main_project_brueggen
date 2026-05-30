import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import numpy as np

import parameters as p
import helping_functions as hf


# =========================================================================
# Custom Tuner für dynamische Datenübergabe
# =========================================================================
class LSTMTimeSeriesTuner(kt.RandomSearch):
    """
    Ein benutzerdefinierter Tuner, der vor jedem Trainingsdurchlauf
    die Sequenzen basierend auf der dynamischen 'seq_length' neu generiert.
    """

    def run_trial(self, trial, df, feature_cols, timestamps, *args, **kwargs):
        hp = trial.hyperparameters

        # Wir fragen den Wert ab, den der Tuner in 'build' registriert hat
        current_seq_length = hp.get('seq_length')

        # Temporär den globalen Parameter spiegeln, damit hf.create_sequences_multivar korrekt arbeitet
        p.SEQ_LENGTH = current_seq_length

        # Daten dynamisch schneiden
        X, y_when, _ = hf.create_sequences_multivar(df, feature_cols, timestamps)

        # Chronologischer Split
        split_idx = int(len(X) * (1 - p.TEST_SPLIT))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_when_train, y_when_test = y_when[:split_idx], y_when[split_idx:]

        # Hyperparameter für die Batch Size abfragen
        current_batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256])

        # kwargs für den Fit-Lauf aktualisieren
        kwargs['batch_size'] = current_batch_size
        kwargs['validation_data'] = (X_test, y_when_test)

        # 'sample_weight' bleibt entfernt, da der gewichtete Loss das übernimmt

        return super(LSTMTimeSeriesTuner, self).run_trial(trial, X_train, y_when_train, *args, **kwargs)


# =========================================================================
# Hypermodell-Funktion
# =========================================================================
def build_ultimate_tuning_model(hp):
    num_features = hp.get('num_features')

    # 'seq_length' wird hier definiert/registriert
    current_seq_length = hp.Int('seq_length', min_value=12, max_value=48, step=12)

    inputs = Input(shape=(current_seq_length, num_features), name="Feature_Input")

    # Suchraum für das Modell
    hp_lstm_1 = hp.Int('lstm_1_units', min_value=16, max_value=128, step=16)
    x = LSTM(units=hp_lstm_1, return_sequences=True)(inputs)

    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=hp_dropout_1)(x)

    hp_lstm_2 = hp.Int('lstm_2_units', min_value=16, max_value=64, step=16)
    x = LSTM(units=hp_lstm_2, return_sequences=False)(x)

    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    x = Dropout(rate=hp_dropout_2)(x)

    x = BatchNormalization()(x)

    hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    dense_when = Dense(units=hp_dense_units, activation='relu', name='Dense_When')(x)
    dense_when = BatchNormalization()(dense_when)

    outputs = Dense(1, activation='sigmoid', name='When_Failure')(dense_when)

    model = Model(inputs=inputs, outputs=outputs)

    # HIER INTEGRATION DER GEWICHTE:
    # Wir fixieren die Basis für den Normalbetrieb (No Failure)
    p.POS_WEIGHT = 1.0

    # Der Tuner sucht das optimale relative Gewicht für die Fehlerklasse (Failure).
    # Werte > 1.0 priorisieren das Finden von Ausfällen (höherer Recall für Failure).
    # Werte < 1.0 priorisieren das Vermeiden von Fehlalarmen (höhere Präzision für No Failure).
    p.NEG_WEIGHT = hp.Float('loss_neg_weight', min_value=1.2, max_value=3.0, step=0.25)

    # Suchraum für Optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=hf.weighted_binary_crossentropy(),  # Nutzt die dynamisch gesetzten Parameter
        metrics=['accuracy', tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )
    return model


# =========================================================================
# Tuning-Pipeline Ausführung
# =========================================================================
if __name__ == "__main__":
    print("1. Loading and preprocessing base dataframe...")
    df, scaler, feature_cols, timestamps = hf.load_and_preprocess_data_all_features()
    num_features = len(feature_cols)

    hp = kt.HyperParameters()
    hp.Fixed('num_features', num_features)

    # Instanziierung des Zeitreihen-Tuners
    tuner = LSTMTimeSeriesTuner(
        hypermodel=build_ultimate_tuning_model,
        hyperparameters=hp,
        objective=kt.Objective("val_pr_auc", direction="max"),
        max_trials=20,
        directory='kt_tuning_ultimate',
        project_name='lstm_ultimate_weighted_v2'  # V2 um den Cache sauber zurückzusetzen
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("\n--- Starting Ultimate Hyperparameter Search (Data + Model + Loss Weights) ---")
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