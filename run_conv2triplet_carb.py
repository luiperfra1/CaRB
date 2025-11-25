# run_conv2triplet_carb.py
from __future__ import annotations

import pathlib
import sys
import subprocess
import time
import csv
from typing import List, Tuple, Dict

PROJECT_ROOT = r"C:/Users/maner/Desktop/ESTUDIOS/conversation2bd_memCorto"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from text2triplets.text2triplet import run_kg, KGConfig, SEXTET_PROMPT_EN  # type: ignore

Triple = Tuple[str, str, str]

CARB_ROOT = pathlib.Path(__file__).resolve().parent
TEST_SENTENCES = CARB_ROOT / "data" / "test.txt"
SYSTEM_OUTPUT = CARB_ROOT / "system_outputs" / "test" / "conv2triplet_tabbed.txt"

RESULTS_CSV = CARB_ROOT / "results" / "model_benchmark.csv"
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

MODELS = (
    #"openai/qwen2.5:72b",
    #"openai/deepseek-llm:67b",
    #"openai/llama3.3:latest",

    "openai/qwen2.5:32b",
    #"openai/deepseek-r1:32b",
    "openai/openthinker:32b",
    #"openai/exaone-deep:32b",

   # "openai/qwen2.5:14b",
    "openai/hermes3:8b",
    #"openai/llama3.1:8b",
    #"openai/cogito:8b",
    #"openai/exaone-deep:7.8b",
    #"openai/mixtral:latest"
)


def extract_triples(model_name: str, text: str) -> List[Triple]:
    cfg_obj = KGConfig(model=model_name)

    sextets = run_kg(
        input_text=text,
        context=SEXTET_PROMPT_EN,
        cfg=cfg_obj,
        drop_invalid=True,
        print_triplets=False,
        sqlite_db_path="./data/users/demo.sqlite",
        reset_log=False,
    ) or []

    # === IMPRIMIR SEXTETAS ===
    print("\n=== SEXTETAS EXTRAÍDAS ===")
    for s in sextets:
        print("  ", s)

    triples: List[Triple] = []

    # === IMPRIMIR TRIPLETAS ===
    print("=== TRIPLETAS (S, V, O) ===")
    for s in sextets:
        if isinstance(s, (list, tuple)) and len(s) >= 3:
            subj = str(s[0]).strip()
            verb = str(s[1]).strip()
            obj = str(s[2]).strip()
            print(f"  ({subj}, {verb}, {obj})")
            triples.append((subj, verb, obj))

    return triples


def run_carb_evaluation() -> Dict[str, float]:
    carb_cmd = [
        sys.executable,
        str(CARB_ROOT / "carb.py"),
        f"--gold=data/gold/test.tsv",
        f"--out=dump/conv2triplet.dat",
        f"--tabbed=system_outputs/test/conv2triplet_tabbed.txt"
    ]

    print("\n=== Ejecutando evaluación CaRB ===")
    print(" ".join(carb_cmd), "\n")

    result = subprocess.run(carb_cmd, capture_output=True, text=True)

    stdout = result.stdout
    print("=== SALIDA CaRB ===")
    print(stdout)

    metrics = {"AUC": 0.0, "precision": 0.0, "recall": 0.0, "F1": 0.0}

    try:
        line = [l for l in stdout.split("\n") if "AUC:" in l][0]
        parts = line.split("Optimal")[0].strip()
        metrics["AUC"] = float(parts.replace("AUC:", "").strip())

        opt = line.split("[")[1].split("]")[0]
        p, r, f1 = [float(x) for x in opt.split()]
        metrics["precision"] = p
        metrics["recall"] = r
        metrics["F1"] = f1

    except Exception:
        pass

    return metrics


def evaluate_model(model_name: str) -> Dict[str, float]:
    print(f"\n============================")
    print(f" Evaluando modelo: {model_name}")
    print("============================\n")

    SYSTEM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_time_no_warmup = 0.0
    num_sentences_no_warmup = 0

    with TEST_SENTENCES.open("r", encoding="utf-8") as fin, \
         SYSTEM_OUTPUT.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            if idx > 100:
                print("\n[INFO] Modo test: procesadas solo las primeras 10 frases.")
                break

            sent = line.strip()
            if not sent:
                continue

            print(f"\n=== [{idx}] TEXTO DE ENTRADA ===")
            print(sent)
            print("================================")

            # Tiempo por frase
            t_start = time.perf_counter()
            triples = extract_triples(model_name, sent)
            t_sent = time.perf_counter() - t_start

            # Ignorar el warm-up de la primera frase
            if idx > 1:
                total_time_no_warmup += t_sent
                num_sentences_no_warmup += 1

            # Guardar tripletas en archivo de CaRB
            for subj, rel, obj in triples:
                fout.write(f"{sent}\t1.0\t{rel}\t{subj}\t{obj}\n")

    # Ejecutar CaRB
    metrics = run_carb_evaluation()
    metrics["time"] = total_time_no_warmup

    # === NUEVO: Imprimir puntuación por modelo ===
    print("\n=== RESULTADO MODELO", model_name, "===")
    print(f"AUC: {metrics['AUC']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['F1']}")
    print(f"Tiempo (sin warmup): {metrics['time']:.3f}s")
    print("=============================================\n")

    return metrics



def main():
    print("==== INICIANDO BENCHMARK MULTIMODELO ====\n")

    results = []

    for model_name in MODELS:
        m = evaluate_model(model_name)
        m["model"] = model_name
        results.append(m)

    # Guardar CSV
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "AUC", "precision", "recall", "F1", "time"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n==== FIN BENCHMARK ====")
    print(f"CSV guardado en: {RESULTS_CSV}\n")


if __name__ == "__main__":
    main()
