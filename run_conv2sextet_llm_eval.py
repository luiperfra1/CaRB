# run_conv2sextet_llm_eval_models.py
from __future__ import annotations

import os
import pathlib
import sys
import time
import json
import csv
import re
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

print(f"[INFO] PROJECT_ROOT={PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path and PROJECT_ROOT is not None:
    sys.path.append(PROJECT_ROOT)

# --- Extractor (usa su pipeline actual de sextetas) ---
from text2triplets.text2triplet import run_kg, KGConfig, SEXTET_PROMPT_EN  # type: ignore

# --- Lógica del juez (externalizada) ---
from eval_llm_juez import (
    Sextet,
    EVALUATOR_MODEL,
    build_evaluator,
    evaluate_sentence,
)

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT / "data"
TEST_SENTENCES = DATA_ROOT / "test.txt"

OUT_ROOT = SCRIPT_ROOT / "results_llm_eval_models"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_SENTENCES = 100 # igual que antes


# ==============================
# MODELOS A PROBAR (extractores)
# ==============================
MODELS = (
    # === MODELOS GRANDES (para pruebas serias pero lentas) ===
    # "openai/qwen2.5:72b",
    # "openai/deepseek-llm:67b",
    # "openai/llama3.3:latest",

    # === MODELOS MEDIOS ===
    "openai/qwen2.5:32b",       # activo por defecto
    # "openai/deepseek-r1:32b",
    # "openai/openthinker:32b",   # contraste útil
    # "openai/exaone-deep:32b",

    # === MODELOS PEQUEÑOS ===
    # "openai/qwen2.5:14b",
    "openai/hermes3:8b",        # baseline pequeño activo
    # "openai/llama3.1:8b",
    # "openai/cogito:8b",
    # "openai/exaone-deep:7.8b",
    # "openai/mixtral:latest"
)


def slugify_model(model_name: str, n_sentences: int) -> str:
    """
    Normaliza el nombre del modelo extractor para crear carpetas limpias.
    Ejemplo:
        "openai/qwen2.5:32b" → "qwen2.5_32b_50s"
    """
    # 1) Quitamos prefijo tipo "openai/"
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    # 2) Convertimos ":" a "_" y eliminamos caracteres raros
    s = model_name.strip().replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

    # 3) Añadimos sufijo "_Ns" (ej: "_50s")
    s += f"_{n_sentences}s"
    return s



def extract_sextets(text: str, extractor_model: str) -> List[Sextet]:
    cfg = KGConfig(model=extractor_model)
    sextets = run_kg(
        input_text=text,
        context=SEXTET_PROMPT_EN,
        cfg=cfg,
        drop_invalid=True,
        print_triplets=False,
        sqlite_db_path="./data/users/demo.sqlite",
        reset_log=False,
    ) or []

    fixed: List[Sextet] = []
    for s in sextets:
        if isinstance(s, (list, tuple)) and len(s) >= 6:
            fixed.append(tuple(str(x).strip() for x in s[:6]))  # type: ignore
        elif isinstance(s, (list, tuple)) and len(s) >= 3:
            ss = [str(x).strip() for x in s]
            while len(ss) < 6:
                ss.append("none")
            fixed.append(tuple(ss[:6]))  # type: ignore
        else:
            continue

    return fixed


def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_for_model(extractor_model: str, evaluator: Any) -> Dict[str, Any]:
    # Número de frases procesadas = MAX_SENTENCES o número real del archivo
    model_slug = slugify_model(extractor_model, MAX_SENTENCES)

    model_out = OUT_ROOT / model_slug
    model_out.mkdir(parents=True, exist_ok=True)

    predictions_json = model_out / "predictions_sextets.json"
    eval_jsonl = model_out / "evaluation_per_sentence.jsonl"
    results_csv = model_out / "results_summary.csv"

    print(f"\n\n==============================")
    print(f"== MODELO EXTRACTOR: {extractor_model}")
    print(f"== MODELO EVALUADOR: {EVALUATOR_MODEL}")
    print(f"== OUT: {model_out}")
    print(f"==============================\n")

    predictions: Dict[str, List[Sextet]] = {}

    extraction_time_no_warmup = 0.0
    n_time = 0

    # -------- EXTRACCIÓN DE SEXTETAS --------
    with TEST_SENTENCES.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, start=1):
            if idx > MAX_SENTENCES:
                print(f"[INFO] Modo test: procesadas solo las primeras {MAX_SENTENCES} frases.")
                break

            sent = line.strip()
            if not sent:
                continue

            print(f"\n=== [{idx}] TEXTO DE ENTRADA ===")
            print(sent)
            print("================================")

            t0 = time.perf_counter()
            sextets = extract_sextets(sent, extractor_model)
            dt = time.perf_counter() - t0

            if idx > 1:
                extraction_time_no_warmup += dt
                n_time += 1

            predictions[sent] = sextets

            print("\n=== SEXTETAS EXTRAÍDAS ===")
            if sextets:
                for s in sextets:
                    print("  ", s)
            else:
                print("  (vacío)")

    with predictions_json.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Predicciones guardadas en: {predictions_json}")

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
                f"\n--- Evaluación [{idx}] --- "
                f"overall={overall} | corr={correctness} comp={completeness} "
                f"noHall={nohall} fmt={fmt} | t={dt:.2f}s"
            )

    summary = {
        "extractor_model": extractor_model,
        "evaluator_model": EVALUATOR_MODEL,
        "n_sentences": len(predictions),
        "avg_overall": avg(overall_scores),
        "avg_correctness": avg(correctness_scores),
        "avg_completeness": avg(completeness_scores),
        "avg_no_hallucination": avg(nohall_scores),
        "avg_format_quality": avg(format_scores),
        "extraction_time_no_warmup_s": extraction_time_no_warmup,
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
    print("==== INICIANDO EXTRACCIÓN -> EVALUACIÓN LLM (MULTI-MODEL) ====\n")

    if not TEST_SENTENCES.exists():
        raise FileNotFoundError(f"No existe {TEST_SENTENCES}. Revise ruta/archivo.")

    # Construimos juez una sola vez (modelo fijo, importado de eval_llm_juez)
    evaluator = build_evaluator()

    all_summaries: List[Dict[str, Any]] = []
    for m in MODELS:
        try:
            summary = run_for_model(m, evaluator)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n[ERROR] Falló modelo {m}: {e}\n")
            # continúa con el resto
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

    print("\n==== FIN MULTI-MODEL ====")


if __name__ == "__main__":
    main()
