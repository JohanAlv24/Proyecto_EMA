import numpy as np
import pandas as pd
from Dynamic_Time_Warping import dtw_distance, compute_dtw_matrix, compute_dtw_test_matrix
def convert_ts(path):
    df_X = pd.read_csv(path+'_TRAIN.tsv', sep="\t")
    train = df_X.values

    df_Y = pd.read_csv(path+'_TEST.tsv', sep="\t")
    test = df_Y.values

    return train, test


if __name__ == "__main__":
    datasets = [
        "ArrowHead",
        "Beef",
        "BirdChicken",
        "CBF",
        "CinCECGTorso",
        "Computers",
        "CricketY",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineCorrect",
        "Earthquakes",
        "ECG5000",
        "ElectricDevices",
        "FaceFour",
        "Fish",
        "FordB",
        "Ham",
        "Herring",
        "InsectWingbeatSound",
        "LargeKitchenAppliances",
        "Lightning7",
        "Meat",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxTW",
        "NonInvasiveFetalECGThorax1",
        "OliveOil",
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
        "Wine",
        "Worms",
        "Yoga"
    ]

    


