import numpy as np
import pandas as pd
from clasificador import ClasificadorSeriesTiempo
from dtaidistance import dtw
import time

def exportar_resultados_excel(lista_resultados, nombre_archivo="resultados.xlsx"):
    filas = []

    for r in lista_resultados:
        fila = {
            # KNN
            "k": r["KNN"]["k"],
            "knn_accuracy_mean": r["KNN"]["accuracy_mean"],
            "knn_accuracy_std": r["KNN"]["accuracy_std"],
            "knn_precision_mean": r["KNN"]["precision_mean"],
            "knn_precision_std": r["KNN"]["precision_std"],
            "knn_recall_mean": r["KNN"]["recall_mean"],
            "knn_recall_std": r["KNN"]["recall_std"],

            # SVM
            "gamma": r["SVM"]["gamma"],
            "svm_accuracy_mean": r["SVM"]["accuracy_mean"],
            "svm_accuracy_std": r["SVM"]["accuracy_std"],
            "svm_precision_mean": r["SVM"]["precision_mean"],
            "svm_precision_std": r["SVM"]["precision_std"],
            "svm_recall_mean": r["SVM"]["recall_mean"],
            "svm_recall_std": r["SVM"]["recall_std"],
        }

        filas.append(fila)

    df = pd.DataFrame(filas)

    # Exportar
    df.to_excel(nombre_archivo, index=False)


def convert_ts(path):
    df_train = pd.read_csv(path + '_TRAIN.tsv', sep="\t", header=None)
    df_test  = pd.read_csv(path + '_TEST.tsv', sep="\t", header=None)

    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values

    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    return X_full, y_full


def NMC(mu, sigma, tau=None):
    n = len(mu)
    
    P = np.zeros((n, 2, 2))
    P[:, 0, 0] = (sigma + mu**2) / sigma
    P[:, 0, 1] = mu / sigma
    P[:, 1, 0] = mu / sigma
    P[:, 1, 1] = 1 / sigma

    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.clip(eigvals, 1e-12, None)

    D_inv_sqrt = np.zeros_like(P)
    D_inv_sqrt[:, 0, 0] = 1 / np.sqrt(eigvals[:, 0])
    D_inv_sqrt[:, 1, 1] = 1 / np.sqrt(eigvals[:, 1])

    P_inv_sqrt = eigvecs @ D_inv_sqrt @ np.transpose(eigvecs, (0, 2, 1))

    M = P_inv_sqrt[:, None] @ P[None, :] @ P_inv_sqrt[:, None]

    eigvals_M = np.linalg.eigh(M)[0]
    eigvals_M = np.clip(eigvals_M, 1e-12, None)

    A = np.sum(np.log(eigvals_M)**2, axis=-1)  # (n, n)

    if tau is not None:
        A = np.where(A > tau, A, 0.0)

    return A


def transform_networks(A, lamb, return_tangent=True):
    """
    A_all: array (K, n, n)  -> matrices de adyacencia A_k
    lam: float > 0
    return_tangent: si True, también retorna proyección en espacio tangente

    Returns:
        L_all: (K, n, n) matrices SPD
        Y_all: (K, n, n) proyecciones (opcional)
        M: (n, n) media SPD (opcional)
    """
    
    K, n, _ = A.shape

    degrees = np.sum(A, axis=2)              # (K, n)
    D = np.zeros_like(A)
    idx = np.arange(n)
    D[:, idx, idx] = degrees

    L = D - A + lamb * np.eye(n)[None, :, :]

    if not return_tangent:
        return L

    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.clip(eigvals, 1e-12, None)

    log_eigvals = np.log(eigvals)

    Log_L = eigvecs @ np.stack([
        np.diag(log_eigvals[k]) for k in range(K)
    ]) @ np.transpose(eigvecs, (0, 2, 1))

    Log_M = np.mean(Log_L, axis=0)

    # exponencial de la media
    eigvals_M, eigvecs_M = np.linalg.eigh(Log_M)
    M = eigvecs_M @ np.diag(np.exp(eigvals_M)) @ eigvecs_M.T

    eigvals_M = np.clip(eigvals_M, 1e-12, None)
    M_inv_sqrt = eigvecs_M @ np.diag(1/np.sqrt(eigvals_M)) @ eigvecs_M.T
    M_sqrt = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.T

    temp = M_inv_sqrt[None, :, :] @ L @ M_inv_sqrt[None, :, :]

    eigvals_t, eigvecs_t = np.linalg.eigh(temp)
    eigvals_t = np.clip(eigvals_t, 1e-12, None)

    log_temp = eigvecs_t @ np.stack([
        np.diag(np.log(eigvals_t[k])) for k in range(K)
    ]) @ np.transpose(eigvecs_t, (0, 2, 1))

    Y = M_sqrt[None, :, :] @ log_temp @ M_sqrt[None, :, :]

    return L, Y, M

if __name__ == "__main__":  
    datasets = [
        "ElectricDevices",
        "ArrowHead",
        "CBF",
        "CinCECGTorso",
        "Computers",
        "CricketY",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineCorrect",
        "Earthquakes",
        "ECG5000",
        "Fish",
        "FordB",
        "Ham",
        "InsectWingbeatSound",
        "LargeKitchenAppliances",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxTW",
        "NonInvasiveFetalECGThorax1",
        "PhalangesOutlinesCorrect",
        "Plane",
        "ProximalPhalanxOutlineCorrect",
        "RefrigerationDevices",
        "ShapeletSim",
        "SmallKitchenAppliances",
        "SonyAIBORobotSurface1",
        "Strawberry",
        "Symbols",
        "ToeSegmentation1",
        "Trace",
        "TwoLeadECG",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryAll",
        "Worms",
        "Yoga"
    ]
    
    classifier_dtw = ClasificadorSeriesTiempo(usar_precomputada=True)
    classifier_euc = ClasificadorSeriesTiempo(usar_precomputada=False)

    metrics_timeseries = []
    metrics_dtw = []
    metrics_latent = []

    
    for data in datasets:
        print(data)
        init = time.time()
        x, y = convert_ts('UCRArchive_2018/'+data+'/'+data)

        print(x.shape)

        D = dtw.distance_matrix_fast(
                                        x,
                                        parallel=True,     
                                        use_c=True,
                                        compact=False         
                                    )
        D = np.nan_to_num(D, nan=0.0, posinf=1e10)
        end = time.time()

        print(end-init)

        init = time.time()
        m_ts = classifier_euc.kfold_tuning(x, y)
        m_ts['Dataset'] = data
        metrics_timeseries.append(m_ts)
    

        m_dtw = classifier_dtw.kfold_tuning(D, y)
        m_dtw['Dataset'] = data
        metrics_dtw.append(m_dtw)
        
        end = time.time()

        print(end-init)
        #m_latent = classifier_euc.kfold_tuning(D, y)
        #m_latent['Dataset'] = data
        #metrics_latent.append(classifier_euc.kfold_tuning(x, y))
    
    exportar_resultados_excel(metrics_timeseries, nombre_archivo="Metricas_series_sin_transformar.xlsx")
    exportar_resultados_excel(metrics_dtw, nombre_archivo="Metricas_dtw.xlsx")
    


        
