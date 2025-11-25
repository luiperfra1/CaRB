# run_conv2sextet_llm_eval_models.py
from __future__ import annotations

import pathlib
import sys
import time
import json
import csv
import re
from typing import List, Tuple, Dict, Any

PROJECT_ROOT = r"C:/Users/maner/Desktop/ESTUDIOS/conversation2bd_memCorto"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Extractor (usa su pipeline actual de sextetas) ---
from text2triplets.text2triplet import run_kg, KGConfig, SEXTET_PROMPT_EN  # type: ignore

# --- Evaluator LLM client ---
from utils.llm_client import LLMClient


Sextet = Tuple[str, str, str, str, str, str]

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_ROOT / "data"
TEST_SENTENCES = DATA_ROOT / "test.txt"

OUT_ROOT = SCRIPT_ROOT / "results_llm_eval_models"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_SENTENCES = 50  # igual que antes


# ==============================
# MODELOS A PROBAR (extractores)
# ==============================
MODELS = (
    # === MODELOS GRANDES (para pruebas serias pero lentas) ===
    #"openai/qwen2.5:72b",
    #"openai/deepseek-llm:67b",
    #"openai/llama3.3:latest",

    # === MODELOS MEDIOS ===
    "openai/qwen2.5:32b",       # activo por defecto
    #"openai/deepseek-r1:32b",
    "openai/openthinker:32b",   # contraste útil
    #"openai/exaone-deep:32b",

    # === MODELOS PEQUEÑOS ===
    #"openai/qwen2.5:14b",
    "openai/hermes3:8b",        # baseline pequeño activo
    #"openai/llama3.1:8b",
    #"openai/cogito:8b",
    #"openai/exaone-deep:7.8b",
    #"openai/mixtral:latest"
)

# ==============================
# MODELO EVALUADOR (fijo)
# ==============================
EVALUATOR_MODEL = "qwen2.5:32b"


EVALUATOR_SYSTEM_PROMPT = """
You are a strict but fair evaluator of extracted sextets from a sentence.

You will receive:
1) ORIGINAL SENTENCE (natural language)
2) EXTRACTED SEXTETS list, each with format:
   (subject, verb, predicate, frequency/temporality, condition, probability)

Your task:
- Judge how well the sextets capture ONLY the facts in the original sentence.
- Penalize hallucinations, wrong subjects/verbs/objects, missing info, wrong temporality/condition.
- Reward correct, complete, and well-scoped sextets.

Return ONLY valid JSON with this schema:
{
  "scores": {
    "correctness": 0-5,
    "completeness": 0-5,
    "no_hallucination": 0-5,
    "format_quality": 0-5
  },
  "overall": 0-5,
  "errors": [
    {
      "type": "hallucination|missing_fact|wrong_relation|wrong_subject|wrong_temporality|wrong_condition|format_error",
      "detail": "short explanation"
    }
  ],
  "suggestions": [
    "short actionable suggestion",
    ...
  ]
}

Rules:
- Be concise.
- If something is uncertain in the sentence, sextets must reflect uncertainty via probability/condition; otherwise penalize.
- If sextets are empty and sentence has facts, correctness/completeness should be low.
- "overall" should reflect your true assessment, not a simple average.
"""


def slugify_model(model_name: str) -> str:
    # openai/qwen2.5:32b -> openai_qwen2.5_32b
    s = model_name.strip().replace("/", "_").replace(":", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
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


def build_evaluator() -> Any:
    if LLMClient is None:
        raise RuntimeError(
            "No pude importar LLMClient. Ajuste el import arriba a la ruta correcta."
        )
    return LLMClient(model=EVALUATOR_MODEL, temperature=0.0)


def evaluate_sentence(evaluator: Any, sentence: str, sextets: List[Sextet]) -> Dict[str, Any]:
    prompt = (
        f"ORIGINAL SENTENCE:\n{sentence}\n\n"
        f"EXTRACTED SEXTETS:\n{json.dumps(sextets, ensure_ascii=False)}\n\n"
        "EVALUATE NOW."
    )

    try:
        raw = evaluator.complete(system=EVALUATOR_SYSTEM_PROMPT, user=prompt)
    except Exception:
        raw = evaluator.chat(
            [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()

    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {
        "scores": {
            "correctness": 0,
            "completeness": 0,
            "no_hallucination": 0,
            "format_quality": 0,
        },
        "overall": 0,
        "errors": [{"type": "format_error", "detail": "Evaluator output not valid JSON"}],
        "suggestions": [],
        "raw_output": text,
    }


def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_for_model(extractor_model: str, evaluator: Any) -> Dict[str, Any]:
    model_slug = slugify_model(extractor_model)
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

    evaluator = build_evaluator()

    all_summaries: List[Dict[str, Any]] = []
    for m in MODELS:
        try:
            summary = run_for_model(m, evaluator)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n[ERROR] Falló modelo {m}: {e}\n")
            # continua con el resto
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
