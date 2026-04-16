import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from Dynamic_Time_Warping import compute_dtw_matrix, compute_dtw_test_matrix


class ClasificadorSeriesTiempo:

    def __init__(self, usar_precomputada=False, n_vecinos=1, gamma_svm=0.01):

        self.usar_precomputada = usar_precomputada
        self.n_vecinos = n_vecinos
        self.gamma_svm = gamma_svm

        if usar_precomputada:
            self.knn = KNeighborsClassifier(n_neighbors=n_vecinos, metric='precomputed')
            self.svm = SVC(kernel='precomputed')
        else:
            self.knn = KNeighborsClassifier(n_neighbors=n_vecinos)
            self.svm = SVC(kernel='rbf')


    def transformar_a_kernel(self, matriz_distancias):
        return np.exp(-self.gamma_svm * matriz_distancias)

    def entrenar(self, X_entrenamiento, y_entrenamiento, X_original=None):

        if self.usar_precomputada:
            self.knn.fit(X_entrenamiento, y_entrenamiento)
            K_entrenamiento = self.transformar_a_kernel(X_entrenamiento)
            self.svm.fit(K_entrenamiento, y_entrenamiento)

        else:
            self.knn.fit(X_entrenamiento, y_entrenamiento)
            self.svm.fit(X_entrenamiento, y_entrenamiento)
            self.arbol.fit(X_entrenamiento, y_entrenamiento)

    def predecir(self, X_prueba, X_original_prueba=None):

        if self.usar_precomputada:
            pred_knn = self.knn.predict(X_prueba)
            K_prueba = self.transformar_a_kernel(X_prueba)
            pred_svm = self.svm.predict(K_prueba)

        else:
            pred_knn = self.knn.predict(X_prueba)
            pred_svm = self.svm.predict(X_prueba)
            pred_arbol = self.arbol.predict(X_prueba)

        return pred_knn, pred_svm

    def _metricas(self, y_true, y_pred):

        acc = accuracy_score(y_true, y_pred)
        error = 1 - acc
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            "accuracy": acc,
            "error": error,
            "precision": precision,
            "recall": recall
        }
    def evaluar(self, X_prueba, y_prueba, X_original_prueba=None):

        pred_knn, pred_svm = self.predecir(X_prueba, X_original_prueba)
        return {
            "KNN": self._metricas(y_prueba, pred_knn),
            "SVM": self._metricas(y_prueba, pred_svm)
        }

def kfold_tuning(self, X, y, lista_k=[1,3,5], lista_gamma=None, n_splits=5):

    if lista_gamma is None:
        if self.usar_precomputada:
            gamma_base = 1 / (np.median(X) + 1e-8)
            lista_gamma = gamma_base * np.array([0.5, 1, 2])
        else:
            lista_gamma = [0.01, 0.1, 1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluar_config(k, gamma):

        acc_knn_folds = []
        acc_svm_folds = []

        for train_idx, test_idx in skf.split(X, y):

            if self.usar_precomputada:
                D_train = X[np.ix_(train_idx, train_idx)]
                D_test  = X[np.ix_(test_idx, train_idx)]

                y_train = y[train_idx]
                y_test  = y[test_idx]

                # --- KNN ---
                knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
                knn.fit(D_train, y_train)
                pred_knn = knn.predict(D_test)

                K_train = np.exp(-gamma * D_train)
                K_test  = np.exp(-gamma * D_test)

                svm = SVC(kernel='precomputed')
                svm.fit(K_train, y_train)
                pred_svm = svm.predict(K_test)

            else:
                # ESPACIO NORMAL
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # --- KNN ---
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                pred_knn = knn.predict(X_test)

                # --- SVM ---
                svm = SVC(kernel='rbf', gamma=gamma)
                svm.fit(X_train, y_train)
                pred_svm = svm.predict(X_test)

            acc_knn_folds.append(accuracy_score(y_test, pred_knn))
            acc_svm_folds.append(accuracy_score(y_test, pred_svm))

        return {
            "k": k,
            "gamma": gamma,
            "knn_score": np.mean(acc_knn_folds),
            "svm_score": np.mean(acc_svm_folds)
        }

    resultados = Parallel(n_jobs=6)(
        delayed(evaluar_config)(k, g)
        for k in lista_k
        for g in lista_gamma
    )

    mejor_knn = max(resultados, key=lambda x: x["knn_score"])
    mejor_svm = max(resultados, key=lambda x: x["svm_score"])


    mejor_knn = max(resultados, key=lambda x: x["knn_mean"])
    mejor_svm = max(resultados, key=lambda x: x["svm_mean"])

    metricas_knn = {
        "k": mejor_knn["k"],
        "accuracy_mean": np.mean(mejor_knn["knn_scores"]),
        "accuracy_std": np.std(mejor_knn["knn_scores"])
    }

    metricas_svm = {
        "gamma": mejor_svm["gamma"],
        "accuracy_mean": np.mean(mejor_svm["svm_scores"]),
        "accuracy_std": np.std(mejor_svm["svm_scores"])
    }

    return {
        "KNN": metricas_knn,
        "SVM": metricas_svm
    }
  