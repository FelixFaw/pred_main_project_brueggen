# main.py
import os

# Unterdrückt die TensorFlow C++ Warnungen (oneDNN, CPU instructions)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from lime import lime_tabular
import shap
import matplotlib.pyplot as plt
import warnings
import parameters as p
import helping_functions as hf


def build_predictive_maintenance_model(input_shape, num_stations, num_reasons):
    # Input layer
    inputs = Input(shape=input_shape, name="Feature_Input")

    # LSTM layers to extract temporal patterns from the data
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Shared dense layer
    shared_dense = Dense(64, activation='relu')(x)

    # Output 1: WHEN (Will a failure occur?)
    out_when = Dense(1, activation='sigmoid', name='When_Failure')(shared_dense)

    # Output 2: WHERE (Dynamische Anpassung gegen Softmax-Warnung)
    if num_stations > 1:
        out_where = Dense(num_stations, activation='softmax', name='Where_Station')(shared_dense)
        loss_where = 'sparse_categorical_crossentropy'
    else:
        out_where = Dense(1, activation='sigmoid', name='Where_Station')(shared_dense)
        loss_where = 'binary_crossentropy'

    # Output 3: WHY (Dynamische Anpassung gegen Softmax-Warnung)
    if num_reasons > 1:
        out_why = Dense(num_reasons, activation='softmax', name='Why_Reason')(shared_dense)
        loss_why = 'sparse_categorical_crossentropy'
    else:
        out_why = Dense(1, activation='sigmoid', name='Why_Reason')(shared_dense)
        loss_why = 'binary_crossentropy'

    # Assemble the model
    model = Model(
        inputs=inputs,
        outputs={
            'When_Failure': out_when,
            'Where_Station': out_where,
            'Why_Reason': out_why
        }
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=p.LEARNING_RATE),
        loss={
            'When_Failure': 'binary_crossentropy',
            'Where_Station': loss_where,
            'Why_Reason': loss_why
        },
        metrics={'When_Failure': 'accuracy', 'Where_Station': 'accuracy', 'Why_Reason': 'accuracy'}
    )
    return model


if __name__ == "__main__":
    print("1. Loading and preprocessing data from Excel files...")
    try:
        df, scaler, station_enc, reason_enc, feature_cols = hf.load_and_preprocess_data_all_features()
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()

    print("2. Creating LSTM sequences...")
    X, y_when, y_where, y_why = hf.create_sequences_multivar(df, feature_cols)
    num_features = len(feature_cols)

    # Chronological train-test split
    split_idx = int(len(X) * (1 - p.TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_when_train, y_when_test = y_when[:split_idx], y_when[split_idx:]
    y_where_train, y_where_test = y_where[:split_idx], y_where[split_idx:]
    y_why_train, y_why_test = y_why[:split_idx], y_why[split_idx:]

    print("3. Computing class weights to counteract data imbalance...")
    classes = np.unique(y_when_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_when_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    print(f"Calculated Weights -> No Failure (0): {class_weight_dict[0]:.2f}, Failure (1): {class_weight_dict[1]:.2f}")

    sample_weights_when = np.array([class_weight_dict[label] for label in y_when_train])
    sample_weights_where = np.ones(len(y_where_train))
    sample_weights_why = np.ones(len(y_why_train))

    sample_weights_dict = {
        'When_Failure': sample_weights_when,
        'Where_Station': sample_weights_where,
        'Why_Reason': sample_weights_why
    }

    print("4. Building and compiling the Multi-Output LSTM model...")
    model = build_predictive_maintenance_model(
        input_shape=(p.SEQ_LENGTH, num_features),
        num_stations=len(station_enc.classes_),
        num_reasons=len(reason_enc.classes_)
    )

    print("5. Starting training with sample weights...")
    history = model.fit(
        X_train,
        {'When_Failure': y_when_train, 'Where_Station': y_where_train, 'Why_Reason': y_why_train},
        validation_data=(
            X_test,
            {'When_Failure': y_when_test, 'Where_Station': y_where_test, 'Why_Reason': y_why_test}
        ),
        epochs=p.EPOCHS,
        batch_size=p.BATCH_SIZE,
        verbose=1,
        sample_weight=sample_weights_dict
    )

    print("6. Plotting training curves...")
    hf.plot_training_history(history)

    print("\n--- 7. MODEL EVALUATION ---")
    predictions = model.predict(X_test)
    pred_when_classes = (predictions['When_Failure'] > 0.5).astype(int).flatten()

    print(classification_report(y_when_test, pred_when_classes, labels=[0, 1], target_names=["No Failure", "Failure"],
                                zero_division=0))

    # =========================================================================
    # LIME Setup
    # =========================================================================
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
            fail_probs = model.predict(reshaped_3d, verbose=0)['When_Failure']
            return np.hstack([1 - fail_probs, fail_probs])


        print(f"Generating LIME explanation for failure instance at index {idx_to_explain}...")
        exp = explainer.explain_instance(
            data_row=instance_3d.reshape(-1),
            predict_fn=lime_predict_wrapper,
            num_features=15,
            labels=(1,)
        )
        exp.save_to_file(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\lime_explanation_failure.html')
        print("LIME Report successfully saved!")
    else:
        print("Note: No machine failures were found in the test dataset for LIME.")

    # =========================================================================
    # SHAP Setup
    # =========================================================================
    print("\n--- 9. Starting SHAP (Shapley Additive exPlanations) ---")
    shap_model = Model(inputs=model.input, outputs=model.get_layer('When_Failure').output)

    background_indices = np.random.choice(X_train.shape[0], 100, replace=False)
    background_data = X_train[background_indices]

    print("Initialisiere SHAP GradientExplainer (kann einen Moment dauern)...")

    # Unterdrücke die harmlosen Warnungen für den SHAP-Prozess
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")
        warnings.filterwarnings("ignore", category=FutureWarning)

        explainer_shap = shap.GradientExplainer(shap_model, background_data)

        if len(failure_indices) > 0:
            test_samples_to_explain = X_test[failure_indices[:20]]
            shap_values = explainer_shap.shap_values(test_samples_to_explain)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values_2d = np.sum(shap_values, axis=1)
            test_samples_2d = np.mean(test_samples_to_explain, axis=1)

            print("Erstelle SHAP Summary Plot...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_2d, test_samples_2d, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(r'C:\Users\tanne\Documents\Hochschule\Brueggen_plots\shap_summary_plot.png')
            plt.close()
            print("SHAP Plot gespeichert unter 'shap_summary_plot.png'.")

    # =========================================================================
    # Permutation Feature Importance
    # =========================================================================
    print("\n--- 10. Starting Permutation Feature Importance ---")
    baseline_preds = (model.predict(X_test, verbose=0)['When_Failure'] > 0.5).astype(int).flatten()
    baseline_acc = np.mean(baseline_preds == y_when_test)
    print(f"Baseline Accuracy (When): {baseline_acc:.4f}")

    feature_importances = {}
    print("Mische Features einzeln durch, um globale Wichtigkeit zu messen...")

    for i, feature_name in enumerate(feature_cols):
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, :, i])

        shuffled_preds = (model.predict(X_test_shuffled, verbose=0)['When_Failure'] > 0.5).astype(int).flatten()
        shuffled_acc = np.mean(shuffled_preds == y_when_test)

        importance = baseline_acc - shuffled_acc
        feature_importances[feature_name] = importance

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
    print("Permutation Importance Plot gespeichert unter 'permutation_importance.png'.")