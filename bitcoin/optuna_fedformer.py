import optuna
import subprocess
import json
import re

# ==== ПАРАМЕТРИ ПОШУКУ ====
N_TRIALS = 20  # кількість експериментів

def objective(trial):
    # 1️⃣ Пропонуємо параметри
    seq_len = trial.suggest_categorical("seq_len", [96, 192, 256, 336])
    label_len = trial.suggest_categorical("label_len", [48, 96, 168])
    pred_len = trial.suggest_categorical("pred_len", [24, 48])
    modes = trial.suggest_categorical("modes", [8, 16, 32, 64])
    d_model = trial.suggest_categorical("d_model", [256, 512])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    dropout = trial.suggest_uniform("dropout", 0.05, 0.2)

    # 2️⃣ Формуємо команду для запуску FEDformer
    cmd = [
        "python", "run_custom.py",
        "--is_training", "1",
        "--task_id", "BTC",
        "--model", "FEDformer",
        "--version", "Fourier",
        "--mode_select", "random",
        "--data", "custom",
        "--root_path", "/content/",
        "--data_path", "BTC-USD.csv",
        "--features", "M",
        "--target", "Close",
        "--freq", "d",
        "--seq_len", str(seq_len),
        "--label_len", str(label_len),
        "--pred_len", str(pred_len),
        "--modes", str(modes),
        "--d_model", str(d_model),
        "--batch_size", str(batch_size),
        "--train_epochs", "10",
        "--patience", "3",
        "--learning_rate", str(lr),
        "--dropout", str(dropout),
        "--use_gpu", "True"
    ]

    print(f"\n🚀 Running trial with params: {cmd}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # 3️⃣ Витягуємо MSE з виводу
        match = re.search(r"mse:([\d\.]+)", result.stdout)
        if match:
            mse = float(match.group(1))
            print(f"✅ Trial MSE: {mse}")
            return mse
        else:
            print("⚠️ MSE not found, assigning inf")
            print(result.stdout)
            return float("inf")

    except subprocess.TimeoutExpired:
        print("❌ Timeout expired")
        return float("inf")
    except Exception as e:
        print(f"❌ Error: {e}")
        return float("inf")


import pandas as pd
import os

def save_optuna_results(study, filename="optuna_results.csv"):
    results = []

    for trial in study.trials:
        row = trial.params.copy()
        row["trial_number"] = trial.number
        row["value"] = trial.value  # objective score (MSE or MAE)
        row["state"] = trial.state.name
        
        # Optional: якщо ти повертаєш окремі метрики з моделі
        if "metrics" in trial.user_attrs:
            row.update(trial.user_attrs["metrics"])
        
        results.append(row)

    df = pd.DataFrame(results)

    # If file exists — append; else — create
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# ==== ЗАПУСК OPTUNA ====
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)
save_optuna_results(study, "BTC_optuna_log.csv")


print("\n🎯 Найкращі параметри:")
print(study.best_params)

# ==== ЗБЕРІГАЄМО РЕЗУЛЬТАТИ ====
with open("/content/FEDformer/results/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

print("✅ Збережено у /content/FEDformer/results/best_params.json")
