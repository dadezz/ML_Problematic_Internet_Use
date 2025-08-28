import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import keras_tuner as kt
import tensorflow as tf


def prepare_X (X_tr, X_val, y_tr, y_val, SEED):
    tf.random.set_seed(SEED)

    scaler = StandardScaler()

    # uso il train per calcolare media e deviazione standard, e standardizzo tutto (train, val, test)

    X_tr_filled = X_tr.fillna(-9999).copy() 
    X_val_filled = X_val.fillna(-9999).copy()  

    X_tr_scaled = scaler.fit_transform(X_tr_filled)    
    X_val_scaled = scaler.transform(X_val_filled)

    classes = np.unique(y_tr) # array delle classi presenti nel training set
    num_classes = len(classes)

    auto_weights = compute_class_weight(class_weight='balanced',
                                        classes=classes,
                                        y=y_tr)
    class_weights = {cls: w for cls, w in zip(classes, auto_weights)}

    return X_tr_scaled, X_val_scaled, class_weights

def build_model(n_features,
                n_classes = 4,
                hidden_units=[64,32],
                activation='relu',
                lr=1e-3,
                dropout=0.0,
                l2_reg=0.0,
                loss_case_binary = "binary_crossentropy",
                loss_case_multiclass = "sparse_categorical_crossentropy",
                loss_case_regression="mse",
                metrics=['accuracy', ]):
    
    # Definisce il layer di input, dimensione = numero di feature
    inputs = layers.Input(shape=(n_features,))
    
    x = inputs    # Variabile temporanea per "costruire" i layer successivi
    for units in hidden_units:
        
        x = layers.Dense(units, # denso -> fully-connected
                         activation = activation,
                         kernel_regularizer = keras.regularizers.l2(l2_reg))(x) # l2_reg
        if dropout and dropout > 0.0: 
            x = layers.Dropout(dropout)(x) # applico il dropout
    if n_classes == None:
        out = layers.Dense(1, activation='linear')(x)
        loss = loss_case_regression
    elif n_classes <= 2: # classificazione binaria
        out = layers.Dense(1, activation='sigmoid')(x)
        loss = loss_case_binary
    else: # classificazione multiclasse
        out = layers.Dense(n_classes, activation='softmax')(x)
        loss = loss_case_multiclass

    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss,
                  metrics=metrics)
    
    return model


def run_nn (X, Xv, y, yv, _num_features = 11, _num_classes = 4, hidden_units=[64,32], activation='relu', lr=1e-4, dropout=0, l2_reg=0, patience_early_stopping = 50, patience_reduce_lr = 6, factor = 0.5, min_lr=1e-6, epochs = 500, batch_size=32, loss_case_binary = "binary_crossentropy", loss_case_multiclass = "sparse_categorical_crossentropy", loss_case_regression="mse", metrics=['accuracy', ], weights = None, classes = ["0","1","2","3"]):
    """"
    * _num_features: int, numero di feature in input
    * _num_classes: int, numero di classi in output (None per regressione)
    * hidden_units: lista di interi, numero di neuroni per layer nascosti
    * activation: stringa, funzione di attivazione da usare nei layer nascosti
    * lr: float, learning rate iniziale
    * dropout: float, percentuale di dropout da applicare
    * l2_reg: float, coefficiente di regolarizzazione L2
    * patience_early_stopping: int, numero di epoche senza miglioramento prima di fermare l'allenamento
    * patience_reduce_lr: int, numero di epoche senza miglioramento prima di ridurre il learning rate
    * factor: float, fattore di riduzione del learning rate se la loss di validazione non migliora
    * min_lr: float, learning rate minimo a cui ridurre
    * epochs: int, numero massimo di epoche per l'allenamento
    * batch_size: int, numero di campioni per batch
    * loss_case_binary: stringa, funzione di loss da usare per il caso binario
    * loss_case_multiclass: stringa, funzione di loss da usare per il caso multiclasse
    * loss_case_regression: stringa, funzione di loss da usare per il caso di regressione
    * metrics: lista di stringhe, metriche da usare per valutare il modello
    * weights: dizionario, pesi delle classi per il bilanciamento
    * classes: lista, nomi delle classi per la confusion matrix
    """
    # ----------------------
    # modellazione effettiva
    # ----------------------
    model = build_model(
        n_features=_num_features,          # input
        n_classes=_num_classes,          # output
        hidden_units=hidden_units,      # due con 64 e 32 neuroni
        activation=activation,          # f di attivazione
        lr=lr,                          # learning rate iniziale
        dropout=dropout,                # dropout inizialmente 0
        l2_reg=l2_reg,                 # regolarizzazione L2 idem
        loss_case_binary=loss_case_binary,  # loss per caso binario
        loss_case_multiclass=loss_case_multiclass,  # loss per caso multiclasse
        loss_case_regression=loss_case_regression,  # loss per caso regressione
        metrics=metrics,                # metriche da usare
    )

    callbacks = [
        # early stop se la loss di validazione non migliora per tot epoche consecutive
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early_stopping, restore_best_weights=True),

        # Se la loss di validazione non migliora per tot epoche, riduce il learning rate del factor%
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience_reduce_lr, min_lr=min_lr)
    ]

    history = model.fit(
        X,
        y,
        validation_data=(Xv, y),
        epochs=epochs, 
        batch_size=batch_size, # numero di campioni per batch
        class_weight=weights, # pesi classi
        callbacks=callbacks, 
        verbose=1
    )

    # ----------------------
    # risultati
    # ----------------------

    print("training terminato")

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss per epoca'); plt.legend()

    # Se è regressione, grafico MAE invece che accuracy
    if _num_classes > 0:
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.title('Accuracy per epoca'); plt.legend()
    else:
        if 'mae' in history.history:
            plt.subplot(1,2,2)
            plt.plot(history.history['mae'], label='train MAE')
            plt.plot(history.history['val_mae'], label='val MAE')
            plt.title('MAE per epoca'); plt.legend()
    plt.show()       

    # Predizioni di probabilità sul validation set
    y_val_pred_proba = model.predict(Xv)

    # Conversione da probabilità a classe predetta
    if _num_classes == None:
        y_val_pred = np.rint(y_val_pred_proba).astype(int).ravel()
        y_val_pred = np.clip(y_val_pred, 0, 3)
    elif _num_classes <= 2:
        # Caso binario: threshold 0.5
        y_val_pred = (y_val_pred_proba.ravel() >= 0.5).astype(int)
    else:
        # Caso multiclasse: argmax
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    

    # --- Report di classificazione ---
    print("Classification report (validation):")
    print(classification_report(yv, y_val_pred, digits=4))

    # --- Confusion matrix ---
    cm = confusion_matrix(yv, y_val_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')        # asse X: predizioni
    plt.ylabel('True')             # asse Y: valori reali
    plt.title('Confusion Matrix (validation)')
    plt.show()

    if _num_classes and _num_classes > 3:
        # --- Recall specifico per classe 3 ---
        rec_severe = recall_score(yv, y_val_pred, labels=[3], average='macro')
        print(f"Recall classe severe (3): {rec_severe:.4f}")

def impute_missing_values(X_tr, X_val, SEED):
    # Usa IterativeImputer con RandomForestRegressor per imputare i valori mancanti
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=SEED),
        max_iter=10,
        random_state=SEED
    )
    X_tr_imputed = imputer.fit_transform(X_tr)
    X_val_imputed = imputer.transform(X_val)
    
    # eseguo quindi la standardizzazione
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    return X_tr_scaled, X_val_scaled

def rf_cascade(X_tr, y_tr, X_val, y_val, SEED):
    # y come classe binaria
    y_bin_tr = np.where(np.isin(y_tr, [0, 1]), 0, 1)
    y_bin_val = np.where(np.isin(y_val, [0, 1]), 0, 1)

    rf = RandomForestClassifier(
        random_state=SEED,
        class_weight='balanced',
        n_estimators=100,
        max_depth=10
    )
    rf.fit(X_tr, y_bin_tr)

    # Predizione RF: probabilità
    rf_probs_train = rf.predict_proba(X_tr)[:,1].reshape(-1,1)
    rf_probs_val = rf.predict_proba(X_val)[:,1].reshape(-1,1)

    #  Predizione RF: classificazione (probabilità discretizzata)
    rf_pred_train = (rf_probs_train >= 0.5).astype(int).ravel()
    rf_pred_val = (rf_probs_val >= 0.5).astype(int).ravel()

    cm_train = confusion_matrix(y_bin_tr, rf_pred_train)
    cm_val = confusion_matrix(y_bin_val, rf_pred_val)

    print("=== Confusion Matrix - TRAIN ===")
    print(cm_train)
    print("\n=== Confusion Matrix - VAL ===")
    print(cm_val)

    print("\n=== Classification Report - TRAIN ===")
    print(classification_report(y_bin_tr, rf_pred_train))
    print("\n=== Classification Report - VAL ===")
    print(classification_report(y_bin_val, rf_pred_val))

    acc_train = accuracy_score(y_bin_tr, rf_pred_train)
    acc_val = accuracy_score(y_bin_val, rf_pred_val)
    print(f"TRAIN Accuracy RF: {acc_train:.4f}")
    print(f"VAL Accuracy RF: {acc_val:.4f}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=['A','B'], yticklabels=['A','B'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix RF - Validation")
    plt.show()

    # aggiungiamo probabilità RF come feature (continua)
    X_bin_tr = np.hstack([X_tr, rf_probs_train])
    X_bin_val = np.hstack([X_val, rf_probs_val])
    n_features_aug = X_bin_tr.shape[1]

    print("Shape train:", X_bin_tr.shape)
    print("Shape val:", X_bin_val.shape)

    return X_bin_tr, X_bin_val, n_features_aug

def tune_hyperparameters(X_tr, y_tr, X_val, y_val, _num_features, _num_classes, weights):
    """
    Funzione che esegue l'hyperparameter tuning usando Keras Tuner.
    """

    def build_model(hp):
        """
        Funzione che costruisce un modello di rete neurale con iperparametri
        definiti da Keras Tuner.
        """

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(_num_features,)))


        # calibra il numero di neuroni nel primo layer denso
        hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=16)
        # calibra la funzione di attivazione
        hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
        # calibra il valore di regolarizzazione L2
        hp_l2 = hp.Float('l2_reg', min_value=0.0000001, max_value=0.01, sampling='log')

        model.add(tf.keras.layers.Dense(units=hp_units_1, activation=hp_activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(hp_l2)))

        # calibra il tasso di Dropout
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(hp_dropout))

        # Aggiungiamo un secondo layer denso, trovando il numero di neuroni migliore
        hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
        model.add(tf.keras.layers.Dense(units=hp_units_2, activation=hp_activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(hp_l2)))

        model.add(tf.keras.layers.Dense(_num_classes, activation='softmax'))

        # --- calibra il Learning Rate dell'ottimizzatore ---
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # Definisce il tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',   # obiettivo: massimizzare accuracy sul validation
        max_trials=100,             # Numero totale di combinazioni da provare
        executions_per_trial=2,     # eseguire due volte il modello serve per questioni di stabilità
        directory='my_keras_tuner', # Cartella dove salvare i risultati
        project_name='dwm_tuning'
    )
        
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    tuner.search(
        X_tr, y_tr,
        epochs=150, #non serve un numero eccessivo, basta che tenda a convergere
        validation_data=(X_val, y_val),
        callbacks=[stop_early],
        class_weight=weights
    )


    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    ---  RICERCA COMPLETATA  ---

    I migliori iperparametri trovati sono:
    - Unità primo layer: {best_hps.get('units_1')}
    - Unità secondo layer: {best_hps.get('units_2')}
    - Funzione di attivazione: {best_hps.get('activation')}
    - Tasso di Dropout: {best_hps.get('dropout'):.2f}
    - Regolarizzazione L2: {best_hps.get('l2_reg'):.4f}
    - Learning Rate: {best_hps.get('learning_rate')}
    """)

    #-------------------------------------------------------------------------------
    # 5. ADDESTRAMENTO DEL MODELLO FINALE CON I PARAMETRI MIGLIORI
    #-------------------------------------------------------------------------------

    # Costruisci il modello con i migliori iperparametri trovati
    final_model = tuner.hypermodel.build(best_hps)


    # Ora addestra questo modello finale su TUTTI i dati di training (training + validation)
    # per dargli più dati possibili prima del test finale.
    X_tr_full = np.concatenate([X_tr, X_val])
    y_tr_full = np.concatenate([y_tr, y_val])

    print("\n💪 Addestramento del modello finale con i parametri ottimali...")

    history = final_model.fit(
        X_tr_full, y_tr_full,
        epochs=500, # Addestra per più epoche, l'early stopping si occuperà di fermarsi
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=1e-6)
        ],
        class_weight=weights,
        verbose=1
    )

    final_model.save('neural_network_tuned_model.h5')

    return final_model