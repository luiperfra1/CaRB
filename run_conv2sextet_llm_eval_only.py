# run_conv2sextet_llm_eval_only.py
from __future__ import annotations

import os
import sys
import json
import csv
import time
import pathlib
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# =======================
#  CONFIG RUTAS / ENTORNO
# =======================

load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

print(f"[INFO] PROJECT_ROOT={PROJECT_ROOT}")
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Importamos la lógica del juez ---
from eval_llm_juez import (  # type: ignore
    Sextet,
    EVALUATOR_MODEL,
    build_evaluator,
    evaluate_sentence,
)

# =======================
#  CONSTANTES DE RUTAS
# =======================

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_ROOT / "results_llm_eval_models"

# MODELOS A EVALUAR (deben tener su carpeta con predictions_sextets.json)
MODELS = (
    "hermes3:8b",
    "qwen2.5:32b",
)
MAX_SENTENCES = 50

# =======================
#  HELPERS
# =======================

def slugify_model(model_name: str, max_sentences: int) -> str:
    """
    Normaliza el nombre del modelo extractor para crear la carpeta de resultados.
    Ejemplos:
      - 'openai/qwen2.5:32b' -> 'qwen2.5_32b_50s'
      - 'qwen2.5:32b'        -> 'qwen2.5_32b_50s'
    """
    # 1) Quitamos prefijo tipo "openai/" si lo hubiera
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    # 2) Sustituimos ':' por '_' y limpiamos caracteres raros
    s = model_name.strip().replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

    # 3) Añadimos sufijo con nº de frases
    s = f"{s}_{max_sentences}s"

    return s



def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_predictions(predictions_json: pathlib.Path) -> Dict[str, List[Sextet]]:
    """
    Carga el JSON de predicciones y lo normaliza a Dict[str, List[Sextet]].
    """
    if not predictions_json.exists():
        raise FileNotFoundError(f"No existe el fichero de predicciones: {predictions_json}")

    with predictions_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    predictions: Dict[str, List[Sextet]] = {}

    for sent, sexts in raw.items():
        fixed: List[Sextet] = []
        for s in sexts:
            if isinstance(s, (list, tuple)) and len(s) >= 6:
                fixed.append(tuple(str(x).strip() for x in s[:6]))  # type: ignore
            elif isinstance(s, (list, tuple)) and len(s) >= 3:
                # Si viniera algo más corto, lo rellenamos como hacías antes
                ss = [str(x).strip() for x in s]
                while len(ss) < 6:
                    ss.append("none")
                fixed.append(tuple(ss[:6]))  # type: ignore
        predictions[sent] = fixed

    return predictions


def run_eval_for_model(extractor_model: str, evaluator: Any) -> Dict[str, Any]:
    """
    Rehace SOLO la evaluación del juez para un modelo extractor dado,
    usando el predictions_sextets.json ya existente.

    Genera:
      - evaluation_per_sentence.jsonl
      - results_summary.csv
    dentro de results_llm_eval_models/<model_slug>/
    """
    
    model_slug = slugify_model(extractor_model, MAX_SENTENCES)
    model_out = OUT_ROOT / model_slug
    model_out.mkdir(parents=True, exist_ok=True)

    predictions_json = model_out / "predictions_sextets.json"
    eval_jsonl = model_out / "evaluation_per_sentence.jsonl"
    results_csv = model_out / "results_summary.csv"

    print(f"\n\n==============================")
    print(f"== MODELO EXTRACTOR: {extractor_model}")
    print(f"== CARPETA: {model_out}")
    print(f"== FICHERO PREDICCIONES: {predictions_json}")
    print(f"== MODELO EVALUADOR: {EVALUATOR_MODEL}")
    print(f"==============================\n")

    # -------- Cargamos predicciones existentes --------
    predictions = load_predictions(predictions_json)

    # -------- EVALUACIÓN CON EL JUEZ LLM --------
    eval_time_no_warmup = 0.0
    n_eval_time = 0

    overall_scores: List[float] = []
    correctness_scores: List[float] = []
    completeness_scores: List[float] = []
    nohall_scores: List[float] = []
    format_scores: List[float] = []

    with eval_jsonl.open("w", encoding="utf-8") as fout:
        for idx, (sent, sextets) in enumerate(predictions.items(), start=1):
            print(f"\n=== [{idx}] Evaluando frase ===")
            print(sent)
            print("================================")

            t0 = time.perf_counter()
            ev = evaluate_sentence(evaluator, sent, sextets)
            dt = time.perf_counter() - t0

            if idx > 1:
                eval_time_no_warmup += dt
                n_eval_time += 1

            scores = ev.get("scores", {}) or {}
            correctness = float(scores.get("correctness", 0))
            completeness = float(scores.get("completeness", 0))
            nohall = float(scores.get("no_hallucination", 0))
            fmt = float(scores.get("format_quality", 0))
            overall = float(ev.get("overall", 0))

            overall_scores.append(overall)
            correctness_scores.append(correctness)
            completeness_scores.append(completeness)
            nohall_scores.append(nohall)
            format_scores.append(fmt)

            record = {
                "sentence": sent,
                "sextets": sextets,
                "evaluation": ev,
                "eval_seconds": dt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"overall={overall} | corr={correctness} comp={completeness} "
                f"noHall={nohall} fmt={fmt} | t={dt:.2f}s"
            )

    # Para mantener el formato del antiguo resumen, rellenamos también extraction_time_no_warmup_s = 0.0
    summary = {
        "extractor_model": extractor_model,
        "evaluator_model": EVALUATOR_MODEL,
        "n_sentences": len(predictions),
        "avg_overall": avg(overall_scores),
        "avg_correctness": avg(correctness_scores),
        "avg_completeness": avg(completeness_scores),
        "avg_no_hallucination": avg(nohall_scores),
        "avg_format_quality": avg(format_scores),
        "extraction_time_no_warmup_s": 0.0,  # no medimos extracción aquí
        "evaluation_time_no_warmup_s": eval_time_no_warmup,
    }

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n==== RESUMEN MODELO ====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("========================\n")

    print(f"[OK] Evaluación por frase: {eval_jsonl}")
    print(f"[OK] Resumen CSV: {results_csv}")

    return summary


def main() -> None:
    print("==== MODO EVAL-ONLY: USANDO PREDICCIONES EXISTENTES ====\n")

    # Construimos el juez una sola vez
    evaluator = build_evaluator()

    all_summaries: List[Dict[str, Any]] = []
    for m in MODELS:
        try:
            summary = run_eval_for_model(m, evaluator)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n[ERROR] Falló modelo {m}: {e}\n")
            continue

    # CSV global agregando todos los modelos
    global_csv = OUT_ROOT / "results_all_models.csv"
    if all_summaries:
        with global_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            for row in all_summaries:
                writer.writerow(row)

        print(f"\n[OK] CSV global con todos los modelos: {global_csv}")

    print("\n==== FIN EVAL-ONLY ====")


if __name__ == "__main__":
    main()
