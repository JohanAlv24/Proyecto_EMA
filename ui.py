"""
Módulo de interfaz de usuario. Contiene todas las funciones interactivas de línea de comandos,
prompts, validación y visualización de resultados.
"""

import os
import csv
import importlib
from typing import Dict, List, Tuple


def get_available_ucr_datasets(archive_path: str) -> List[str]:
    """Detecta carpetas válidas de datasets UCR dentro de archive_path."""
    if not os.path.isdir(archive_path):
        return []

    names: List[str] = []
    for entry in sorted(os.listdir(archive_path)):
        folder = os.path.join(archive_path, entry)
        if not os.path.isdir(folder):
            continue
        train_file = os.path.join(folder, f"{entry}_TRAIN.tsv")
        test_file = os.path.join(folder, f"{entry}_TEST.tsv")
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            names.append(entry)
    return names


def get_archive_path_and_datasets() -> Tuple[str, List[str]]:
    """Solicita la ruta del archivo UCR y devuelve la ruta y datasets disponibles."""
    while True:
        archive_path = prompt_text("Ruta al archivo UCR", "UCRArchive_2018")
        available = get_available_ucr_datasets(archive_path)
        if available:
            print(f"✓ Se encontraron {len(available)} datasets en {archive_path}")
            return archive_path, available
        print(f"✗ No se encontraron datasets válidos en: {archive_path}")
        print("  Verifica que la ruta sea correcta y contenga las carpetas del archive UCR.")


def validate_datasets(selected: List[str], available: List[str]) -> None:
    """Valida que los datasets seleccionados existan en el archivo UCR."""
    missing = [d for d in selected if d not in available]
    if missing:
        raise ValueError(
            "Datasets no encontrados en el archivo UCR: "
            + ", ".join(missing)
            + "\nUsa --list-datasets para ver opciones válidas."
        )


def prompt_text(prompt: str, default: str = "") -> str:
    """Solicita un texto al usuario con valor por defecto opcional."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def prompt_bool(prompt: str, default: bool) -> bool:
    """Solicita una respuesta booleana (s/n) al usuario."""
    default_text = "s" if default else "n"
    while True:
        raw = input(f"{prompt} [s/n] (default: {default_text}): ").strip().lower()
        if not raw:
            return default
        if raw in {"s", "si", "sí", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Entrada inválida. Escribe 's' o 'n'.")


def prompt_int(
    prompt: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Solicita un número entero al usuario con validación opcional de rango."""
    while True:
        raw = prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Entrada inválida. Debe ser un número entero.")
            continue
        if min_value is not None and value < min_value:
            print(f"Entrada inválida. Debe ser >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Entrada inválida. Debe ser <= {max_value}.")
            continue
        return value


def prompt_float(
    prompt: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Solicita un número flotante al usuario con validación opcional de rango."""
    while True:
        raw = prompt_text(prompt, str(default))
        try:
            value = float(raw)
        except ValueError:
            print("Entrada inválida. Debe ser un número.")
            continue
        if min_value is not None and value < min_value:
            print(f"Entrada inválida. Debe ser >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Entrada inválida. Debe ser <= {max_value}.")
            continue
        return value


def prompt_datasets(available: List[str]) -> Tuple[List[str], bool]:
    """Interfaz interactiva para seleccionar datasets de la lista disponible."""
    print("\nDatasets disponibles:")
    for idx, name in enumerate(available, start=1):
        print(f"  {idx:>3}. {name}")

    while True:
        raw = input(
            "\nSelecciona datasets por número o nombre separados por coma, "
            "o escribe 'all' para todos: "
        ).strip()
        if not raw:
            print("Debes seleccionar al menos un dataset.")
            continue

        if raw.lower() in {"all", "todos", "*"}:
            return available, True

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens:
            print("Debes seleccionar al menos un dataset.")
            continue

        selected: List[str] = []
        ok = True
        for token in tokens:
            if token.isdigit():
                pos = int(token)
                if pos < 1 or pos > len(available):
                    print(f"Índice fuera de rango: {token}")
                    ok = False
                    break
                name = available[pos - 1]
            else:
                matches = [n for n in available if n.lower() == token.lower()]
                if not matches:
                    print(f"Dataset no reconocido: {token}")
                    ok = False
                    break
                name = matches[0]
            if name not in selected:
                selected.append(name)

        if not ok:
            continue

        return selected, False


def collect_interactive_config(available: List[str]) -> Dict[str, object]:
    """Recopila la configuración completa del experimento mediante interfaz interactiva."""
    print("\n=== Configuración interactiva del experimento ===")

    datasets, run_all = prompt_datasets(available)
    validate_datasets(datasets, available)

    use_test = prompt_bool(
        "¿Incluir split TEST en los datos de entrenamiento?",
        default=False,
    )

    stratified_kfold = False
    kfold_splits = 3
    if use_test:
        stratified_kfold = prompt_bool(
            "¿Usar StratifiedKFold para dividir TRAIN+TEST?",
            default=True,
        )
        if stratified_kfold:
            kfold_splits = prompt_int(
                "Número de folds para StratifiedKFold",
                3,
                min_value=2,
            )

    paper_split = False
    if datasets == ["ECG5000"]:
        paper_split = prompt_bool(
            "¿Usar protocolo del paper para ECG5000 (4500 train + 500 red latente)?",
            default=False,
        )

    if run_all:
        proceed = prompt_bool(
            "Modo TODOS puede tardar bastante. ¿Deseas continuar?",
            default=False,
        )
        if not proceed:
            raise KeyboardInterrupt("Ejecución cancelada por el usuario.")

    return {
        "datasets": datasets,
        "run_all": run_all,
        "use_test": use_test,
        "stratified_kfold": stratified_kfold,
        "kfold_splits": kfold_splits,
        "paper_ecg5000_split": paper_split,
        "target_length": prompt_int(
            "Longitud maxima de serie (0 = longitud natural de cada dataset)", 0, min_value=0
        ),
        "hidden_size": prompt_int("Unidades ocultas LSTM (hidden_size)", 96, min_value=1),
        "num_layers": prompt_int("Capas LSTM (num_layers)", 1, min_value=1),
        "beta": prompt_float("Peso KL beta", 1.0, min_value=1e-12),
        "epochs": prompt_int("Número de épocas", 100, min_value=1),
        "batch_size": prompt_int("Batch size", 64, min_value=1),
        "val_split": prompt_float("Fracción de validación (val_split)", 0.1, min_value=0.0, max_value=0.9),
        "learning_rate": prompt_float("Learning rate", 5e-4, min_value=1e-12),
        "early_stopping_patience": prompt_int("Early stopping patience (0 desactiva)", 0, min_value=0),
        "seed": prompt_int("Semilla (seed)", 42, min_value=0),
        "device": prompt_text("Dispositivo [auto/cpu/cuda]", "auto").lower(),
        "save_model": prompt_text("Ruta para guardar modelo (.pt), vacío para no guardar", ""),
    }


def print_results_summary(results: List[Dict[str, object]]) -> None:
    """Imprime tabla de métricas por dataset y promedio global."""
    import numpy as np

    col = 22
    sep = "=" * 80
    print(f"\n{sep}")
    print("RESUMEN FINAL  --  un modelo VAE por dataset")
    print(sep)
    print(
        f"  {'Dataset':<{col}} {'N_Train':>8} {'N_Test':>7}"
        f" {'sMAPE_Train%':>13} {'sMAPE_Test%':>12}"
        f" {'MAE_Train':>12} {'MAE_Test':>11}"
        f" {'RMSE_Train':>12} {'RMSE_Test':>11}"
    )
    print("-" * 80)
    valid: List[Dict[str, object]] = []
    for r in results:
        if "error" in r:
            print(f"  {str(r['name']):<{col}}  ERROR: {r['error']}")
        else:
            print(
                f"  {str(r['name']):<{col}} {r['n_train']:>8} {r['n_test']:>7}"
                f" {float(r['smape_train']):>13.4f} {float(r['smape_test']):>12.4f}"
                f" {float(r['mae_train']):>12.6f} {float(r['mae_test']):>11.6f}"
                f" {float(r['rmse_train']):>12.6f} {float(r['rmse_test']):>11.6f}"
            )
            valid.append(r)
    if len(valid) > 1:
        n = len(valid)
        avg_smape_train = float(np.mean([float(r["smape_train"]) for r in valid]))
        avg_smape_test = float(np.mean([float(r["smape_test"]) for r in valid]))
        avg_mae_train = float(np.mean([float(r["mae_train"]) for r in valid]))
        avg_mae_test = float(np.mean([float(r["mae_test"]) for r in valid]))
        avg_rmse_train = float(np.mean([float(r["rmse_train"]) for r in valid]))
        avg_rmse_test = float(np.mean([float(r["rmse_test"]) for r in valid]))
        label = f"PROMEDIO ({n} datasets)"
        print("-" * 80)
        print(
            f"  {label:<{col}} {'':>8} {'':>7}"
            f" {avg_smape_train:>13.4f} {avg_smape_test:>12.4f}"
            f" {avg_mae_train:>12.6f} {avg_mae_test:>11.6f}"
            f" {avg_rmse_train:>12.6f} {avg_rmse_test:>11.6f}"
        )
    print(sep)


def save_mae_test_data_and_plot(results: List[Dict[str, object]], save_model: str) -> None:
    """Guarda MAE de test/train por dataset y grafica ambas métricas vs series de entrenamiento."""
    valid = [r for r in results if "error" not in r and "mae_test" in r]
    if not valid:
        print("No hay resultados válidos para guardar/graficar MAE_test.")
        return

    if save_model.lower().endswith(".pt"):
        out_dir = os.path.dirname(os.path.abspath(save_model)) or os.getcwd()
    else:
        out_dir = os.path.abspath(save_model)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in valid:
        n_train_used = int(r.get("n_train_used", r["n_train"]))
        rows.append(
            {
                "dataset": str(r["name"]),
                "n_train_used": n_train_used,
                "mae_train": float(r["mae_train"]),
                "mae_test": float(r["mae_test"]),
            }
        )

    csv_path = os.path.join(out_dir, "mae_test_por_dataset.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "n_train_used", "mae_train", "mae_test"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Datos MAE (train/test) guardados en: {csv_path}")

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo generar la gráfica (matplotlib no disponible): {exc}")
        return

    rows_sorted = sorted(rows, key=lambda d: d["n_train_used"])
    x = [d["n_train_used"] for d in rows_sorted]
    y_train = [d["mae_train"] for d in rows_sorted]
    y_test = [d["mae_test"] for d in rows_sorted]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y_train, marker="o", label="MAE_train")
    plt.plot(x, y_test, marker="o", label="MAE_test")
    plt.xlabel("Número de series de entrenamiento")
    plt.ylabel("MAE")
    plt.title("MAE (train/test) vs número de series de entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "mae_vs_n_train.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada en: {fig_path}")
