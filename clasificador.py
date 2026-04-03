from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

        self.arbol = DecisionTreeClassifier()

    def transformar_a_kernel(self, matriz_distancias):
        return np.exp(-self.gamma_svm * matriz_distancias)

    def entrenar(self, X_entrenamiento, y_entrenamiento, X_original=None):

        if self.usar_precomputada:
            self.knn.fit(X_entrenamiento, y_entrenamiento)

            K_entrenamiento = self.transformar_a_kernel(X_entrenamiento)
            self.svm.fit(K_entrenamiento, y_entrenamiento)

            if X_original is None:
                raise ValueError("Se requiere X_original para el árbol.")
            self.arbol.fit(X_original, y_entrenamiento)

        else:
            self.knn.fit(X_entrenamiento, y_entrenamiento)
            self.svm.fit(X_entrenamiento, y_entrenamiento)
            self.arbol.fit(X_entrenamiento, y_entrenamiento)

    def predecir(self, X_prueba, X_original_prueba=None):

        if self.usar_precomputada:
            pred_knn = self.knn.predict(X_prueba)

            K_prueba = self.transformar_a_kernel(X_prueba)
            pred_svm = self.svm.predict(K_prueba)

            if X_original_prueba is None:
                raise ValueError("Se requiere X_original_prueba para el árbol.")
            pred_arbol = self.arbol.predict(X_original_prueba)

        else:
            pred_knn = self.knn.predict(X_prueba)
            pred_svm = self.svm.predict(X_prueba)
            pred_arbol = self.arbol.predict(X_prueba)

        return pred_knn, pred_svm, pred_arbol

    def evaluar(self, X_prueba, y_prueba, X_original_prueba=None):

        pred_knn, pred_svm, pred_arbol = self.predecir(X_prueba, X_original_prueba)

        return {
            "KNN": 1 - accuracy_score(y_prueba, pred_knn),
            "SVM": 1 - accuracy_score(y_prueba, pred_svm),
            "Arbol": 1 - accuracy_score(y_prueba, pred_arbol)
        }