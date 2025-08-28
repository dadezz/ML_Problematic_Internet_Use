"""
modulo per la pulizia e preparazione dei dati
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import clone
from sklearn.model_selection import cross_val_score

def season_map(X):
    season_map = {
        # il mapping darà NaN quando incontra NaN, quindi viene gestito in automatico
        'Winter': 0,
        'Spring': 1,
        'Summer': 2,
        'Fall': 3
    }

    # colonne che terminano in "-Season"
    season_cols = [col for col in X.columns if col.endswith('Season')]
    for col in season_cols:
        X[col] = X[col].map(season_map)
    return X

def prepare_data(src = "../data/train.csv", SEED = 42):
    np.random.seed(SEED) # lo faccio adesso ed è globale
    data = pd.read_csv(src) # dataset intero
    Y = data['sii']     # classe da predire
    X = data.drop(columns=['id','sii'])     # features
    # non ha senso lasciare righe in cui la label non si conosce. è supervised learning: ogni riga deve avere la sua classificazione
    X = X[Y.notnull()]
    Y = Y[Y.notnull()]

    # slip train & test + train & val
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    X_train,X_val , y_train, y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=SEED)

    num_of_instances , num_of_columns = X_train.shape
    columns_to_keep = X_train.isnull().sum().sort_values(ascending=False)  < (num_of_instances//2)
    X_train = X_train.loc[:, columns_to_keep] # tolgo le colonne con più del 50% di NaN

    X_train = season_map(X_train)

    return X_train, y_train, X_val, y_val, X_test, y_test

def correlation_matrix(X, y):
    # rimetto l'sii come prima colonna per calcolare la correlazione
    X_temp = X.copy()
    X_temp.insert(0, column="Sii", value=y)

    # Calcolo la matrice di correlazione e guardo la corr col target
    corr_matrix = X_temp.corr().abs()
    target_corr = corr_matrix.loc["Sii"].drop("Sii")

    # Riduco la matrice alle sole feature utili.
    useful_features = target_corr[(target_corr >= 0.1) & (target_corr <= 0.9)].index.tolist()
    corr_matrix_useful = corr_matrix.loc[useful_features, useful_features]

    # Calcolo solo triangolare superiore per evitare duplicati
    upper_tri = corr_matrix_useful.where(np.triu(np.ones(corr_matrix_useful.shape), k=1).astype(bool))

    # Elimino le feature ridondanti (correlazione > 0.9)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    selected_features = [col for col in useful_features if col not in to_drop]
    X_train_filtered = X[selected_features]

    # disegno la heatmap
    f, ax = plt.subplots(figsize=(10, 8))
    corr = X_train_filtered.corr()
    sns.heatmap(corr,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0, vmax=1.0,
        square=True, ax=ax)
    return X_train_filtered

def feature_importance_and_mutual_info_originale(X, y, SEED):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    importances_list = []

    for train_idx, val_idx in kf.split(X, y):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        
        model = RandomForestClassifier(random_state=SEED)
        model.fit(X_fold_train, y_fold_train)
        
        importances_list.append(model.feature_importances_)

    # Media delle importanze sui fold
    mean_importances = np.mean(importances_list, axis=0)

    # Associa nomi feature
    feat_importance_cv = pd.Series(mean_importances, index=X.columns)
    feat_importance_cv_sorted = feat_importance_cv.sort_values(ascending=False)

    # Visualizza le feature importances nel grafico di sinistra
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_importance_cv_sorted, y=feat_importance_cv_sorted.index, palette="viridis")
    plt.title("Feature Importances (RandomForest + Cross-Validation)")
    plt.xlabel("Average Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    # Calcolo Mutual Information
    X_train_mi = X.copy()
    Y_train_mi = y.copy() 

    mi = mutual_info_classif(X_train_mi.fillna(-9999), Y_train_mi)
    mi_series = pd.Series(mi, index=X_train_mi.columns)
    nan_ratio = X_train_mi.isnull().mean()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=nan_ratio, y=mi_series)
    plt.xlabel("NaN ratio")
    plt.ylabel("Mutual Information con la classe")
    plt.title("Feature utility vs missing values")
    plt.grid(True)
    plt.show()

def feature_importance_and_mutual_info(X, y, SEED):

    # inizio con il calcolo delle feature importances
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    importances_list = []

    for train_idx, val_idx in kf.split(X, y):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        model = RandomForestClassifier(random_state=SEED)
        model.fit(X_fold_train, y_fold_train)
        importances_list.append(model.feature_importances_)

    # Media delle importanze sui fold
    mean_importances = np.mean(importances_list, axis=0)

    # Associa nomi feature
    feat_importance_cv = pd.Series(mean_importances, index=X.columns)
    feat_importance_cv_sorted = feat_importance_cv.sort_values(ascending=False)

    # Calcolo Mutual Information
    X_train_mi = X.copy()
    Y_train_mi = y.copy() 
    mi = mutual_info_classif(X_train_mi.fillna(-9999), Y_train_mi)
    mi_series = pd.Series(mi, index=X_train_mi.columns)
    nan_ratio = X_train_mi.isnull().mean()

    # Creo i due grafici affiancati
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # SX: feature importances
    sns.barplot(
        x=feat_importance_cv_sorted,
        y=feat_importance_cv_sorted.index,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Feature Importances (RandomForest + CV)")
    axes[0].set_xlabel("Average Importance Score")
    axes[0].set_ylabel("Features")

    # DX: mutual information vs NaN ratio
    sns.scatterplot(
        x=nan_ratio,
        y=mi_series,
        ax=axes[1]
    )
    axes[1].set_xlabel("NaN ratio")
    axes[1].set_ylabel("Mutual Information con la classe")
    axes[1].set_title("Feature utility vs missing values")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    nan_1 = mi_series[nan_ratio > 0.25]
    print("Feature con NaN ratio > 0.25:")
    print(nan_1)

    mi_high_sorted = mi_series[mi_series > 0.2].sort_values(ascending=False)
    print("\nFeature con alta mutual information, ordinate in modo decrescente:")
    print(mi_high_sorted)
    return feat_importance_cv, feat_importance_cv_sorted

def augment_data(src = "../data/train.csv", dst = "../data/", SEED = 42):

    data = pd.read_csv(src) # dataset intero
    Y = data['sii']     # classe da predire
    X = data.drop(columns=['id','sii'])     # features

    proxy_features = [
        'PCIAT-PCIAT_Total', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 
        'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05',
        'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 
        'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 
        'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 
        'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 
        'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 'PCIAT-Season'
    ]

    # Divisione train/test (80/20)
    # Manteniamo X_test e y_test separati e non li tocchiamo più 
    # fino alla valutazione finale
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=SEED)

    # rieseguo il mapping stagioni sul nuovo dataset
    X_train_full = season_map(X_train_full)

    # nel test devo eliminare i nan (non posso imputarli)
    mask = ~y_test.isna()
    y_test = y_test[mask]
    X_test = X_test[mask]

    # Creiamo un unico dataframe di training per manipolarlo più facilmente
    train_df = pd.concat([X_train_full, y_train_full], axis=1)

    # divisione del training set tra con_sii e senza_sii
    train_con_sii = train_df[train_df['sii'].notnull()].copy()
    train_da_imputare = train_df[train_df['sii'].isnull()].copy()

    print(f"Righe di X_training con 'sii' noto: {len(train_con_sii)}")
    print(f"Righe di X_training da imputare: {len(train_da_imputare)}")

    # Prepariamo i dati per allenare il modello di imputazione
    X_proxy_train = train_con_sii[proxy_features]
    y_proxy_train = train_con_sii['sii']

    # alleniamo quindi una rf per imputazione
    proxy_model = RandomForestClassifier(random_state=SEED)
    proxy_model.fit(X_proxy_train, y_proxy_train)

    # Prepariamo le feature delle righe da imputare
    X_proxy_predict = train_da_imputare[proxy_features]

    # Usiamo il modello per predire i valori di 'sii' mancanti
    sii_imputati = proxy_model.predict(X_proxy_predict)

    # Assegniamo i valori predetti alla colonna 'sii' del set da imputare
    train_da_imputare['sii'] = sii_imputati

    # Uniamo i due set
    train_df_augmented = pd.concat([train_con_sii, train_da_imputare])

    # Ora possiamo separare di nuovo le feature e la label (X e Y)
    y_train_final = train_df_augmented['sii']
    X_train_final = train_df_augmented.drop(columns=['sii'])

    # a questo punto abbiamo un training set completo con 'sii' noto per tutte le righe
    # Possiamo quindi eliminare le feature usate per l'imputazione, così da non influenzare il modello finale
    # con feature che soffrono di data leakage
    X_train_final = X_train_final.drop(columns=proxy_features)

    # Salviamo i dataset finali
    X_train_final.to_csv(f"{dst}X_train_final.csv", index=False)
    y_train_final.to_csv(f"{dst}y_train_final.csv", index=False)
    X_test.to_csv(f"{dst}X_test_final.csv", index=False)
    y_test.to_csv(f"{dst}y_test_final.csv", index=False)
    print("Dataset salvati in formato CSV nella cartella 'data/'.")
    return X_train_final, y_train_final, X_test, y_test

def backward_elimination(X, y, feat_importance_cv, feat_importance_cv_sorted, SEED=42, dst = "../data/"):
    X_be = X.copy()
    y_be = y.copy()
    features = list(feat_importance_cv_sorted.index)

    base_model = RandomForestClassifier(random_state=SEED)
    scores, n_features = [], []

    # Loop backward
    while len(features) > 1:
        X_current = X_be[features]
        score = cross_val_score(clone(base_model), X_current, y_be, cv=5, scoring='accuracy').mean()
        scores.append(score)
        n_features.append(len(features))

        feat_importance_sub = feat_importance_cv[features]
        worst_feature = feat_importance_sub.idxmin()
        features.remove(worst_feature)

    # Calcolo RMSE
    rmse = []
    for f in range(1, len(feat_importance_cv_sorted)):
        rf_small = RandomForestClassifier(random_state=SEED)
        scores = cross_val_score(
            rf_small,
            X[feat_importance_cv_sorted.index[:f]],
            y,
            cv=5,
            scoring='neg_root_mean_squared_error'
        )
        rmse.append(-scores.mean())

    # === Grafici affiancati ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SX: andamento accuracy
    axes[0].plot(n_features, scores, marker='o')
    axes[0].set_xlabel("Numero di feature")
    axes[0].set_ylabel("Accuracy (CV)")
    axes[0].set_title("Backward Feature Elimination (Accuracy)")
    axes[0].invert_xaxis()
    axes[0].grid(True)

    # DX: andamento RMSE
    axes[1].plot(range(1, len(feat_importance_cv_sorted)), rmse, 'o-', label="RMSE")
    axes[1].set_title("RMSE on varying features")
    axes[1].set_xlabel("Number of Best features used")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Top 11 feature
    top11_features = feat_importance_cv_sorted.index[:11].tolist()
    print("Top 11 features (RandomForest CV):")
    for i, f in enumerate(top11_features, 1):
        print(f"{i}. {f}")

    X_be =  X_be[top11_features].copy()
    X_be.to_csv(f"{dst}X_train_final.csv", index=False)
    print("'data/X_train_final.csv' sovrascritto coi nuovi dati.")
    return X_be
