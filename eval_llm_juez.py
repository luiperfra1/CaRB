# eval_llm_juez.py
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# =======================
# CONFIG RUTAS / IMPORTS
# =======================

# 1) Carga el .env local de CaRB (si existe)
load_dotenv()

# 2) Obtiene PROJECT_ROOT (de ese .env o de variables del sistema)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

# 3) Si hay PROJECT_ROOT, añade al sys.path y carga también su .env
if PROJECT_ROOT:
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    root_env = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(root_env):
        # Cargar también el .env del proyecto base donde tienes LLAMUS_URL/OPENAI_API_BASE
        load_dotenv(root_env, override=False)

from utils.llm_client import LLMClient  # type: ignore

Sextet = Tuple[str, str, str, str, str, str]

# Puedes sobreescribir el modelo vía variable de entorno EVALUATOR_MODEL
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "hermes3:8b")


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


# =======================
#  API DEL JUEZ (REUTILIZABLE)
# =======================

def build_evaluator(model: str | None = None) -> Any:
    """
    Construye el cliente LLM del juez.

    Si 'model' es None, usa EVALUATOR_MODEL (o la variable de entorno EVALUATOR_MODEL).
    """
    if LLMClient is None:
        raise RuntimeError(
            "No pude importar LLMClient. Ajusta el import en eval_llm_juez.py."
        )

    model_name = model or EVALUATOR_MODEL
    return LLMClient(model=model_name, temperature=0.0)


def _parse_evaluator_output(raw: Any) -> Dict[str, Any]:
    """
    Intenta parsear la salida del LLM a JSON con el esquema esperado.
    Aplica las mismas heurísticas que tenías en el script original.
    """
    # Si ya es dict, lo devolvemos tal cual
    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()

    # Intento directo de JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Heurística: buscar el primer '{' y el último '}'
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass

    # Fallback si nada funciona
    return {
        "scores": {
            "correctness": 0,
            "completeness": 0,
            "no_hallucination": 0,
            "format_quality": 0,
        },
        "overall": 0,
        "errors": [
            {
                "type": "format_error",
                "detail": "Evaluator output not valid JSON",
            }
        ],
        "suggestions": [],
        "raw_output": text,
    }


def evaluate_sentence(
    evaluator: Any,
    sentence: str,
    sextets: List[Sextet],
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    """
    Evalúa una frase + lista de sextetas con el juez LLM.

    - evaluator: instancia de LLMClient (o compatible) creada con build_evaluator().
    - sentence: frase original.
    - sextets: lista de sextetas [(subj, verb, pred, freq, cond, prob), ...].
    - system_prompt: permite sobreescribir el prompt del juez si quieres;
      si es None, usa EVALUATOR_SYSTEM_PROMPT.
    """
    prompt = (
        f"ORIGINAL SENTENCE:\n{sentence}\n\n"
        f"EXTRACTED SEXTETS:\n{json.dumps(sextets, ensure_ascii=False)}\n\n"
        "EVALUATE NOW."
    )

    sys_prompt = system_prompt or EVALUATOR_SYSTEM_PROMPT

    # Mantenemos la lógica de fallback complete/chat
    try:
        raw = evaluator.complete(system=sys_prompt, user=prompt)
    except Exception:
        raw = evaluator.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    return _parse_evaluator_output(raw)


__all__ = [
    "Sextet",
    "EVALUATOR_MODEL",
    "EVALUATOR_SYSTEM_PROMPT",
    "build_evaluator",
    "evaluate_sentence",
]


# =======================
#  MODO SCRIPT (CLI)
# =======================

def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_batch_eval(
    predictions_path: str,
    out_jsonl_path: str,
    model: str | None = None,
    max_sentences: int | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta el juez sobre un fichero de predicciones y guarda un JSONL.

    Formato esperado de predictions_path (como en tu benchmark):
        {
          "sentence 1": [
              ["subj", "verb", "pred", "freq", "cond", "prob"],
              ...
          ],
          "sentence 2": ...
        }
    """

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions_raw = json.load(f)

    # Normalizamos a Dict[str, List[Sextet]]
    predictions: Dict[str, List[Sextet]] = {}
    for sent, sexts in predictions_raw.items():
        fixed: List[Sextet] = []
        for s in sexts:
            if isinstance(s, (list, tuple)) and len(s) >= 6:
                fixed.append(tuple(str(x).strip() for x in s[:6]))  # type: ignore
        predictions[sent] = fixed

    evaluator = build_evaluator(model=model)

    overall_scores: List[float] = []
    corr_scores: List[float] = []
    comp_scores: List[float] = []
    nohall_scores: List[float] = []
    fmt_scores: List[float] = []

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)

    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for idx, (sent, sexts) in enumerate(predictions.items(), start=1):
            if max_sentences is not None and idx > max_sentences:
                print(f"[INFO] max_sentences={max_sentences} alcanzado, paro aquí.")
                break

            print(f"\n=== [{idx}] Evaluando frase ===")
            print(sent)
            print("================================")

            t0 = time.perf_counter()
            ev = evaluate_sentence(evaluator, sent, sexts)
            dt = time.perf_counter() - t0

            scores = ev.get("scores", {}) or {}
            corr = float(scores.get("correctness", 0))
            comp = float(scores.get("completeness", 0))
            nohall = float(scores.get("no_hallucination", 0))
            fmt = float(scores.get("format_quality", 0))
            overall = float(ev.get("overall", 0))

            overall_scores.append(overall)
            corr_scores.append(corr)
            comp_scores.append(comp)
            nohall_scores.append(nohall)
            fmt_scores.append(fmt)

            record = {
                "sentence": sent,
                "sextets": sexts,
                "evaluation": ev,
                "eval_seconds": dt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"overall={overall} | corr={corr} comp={comp} "
                f"noHall={nohall} fmt={fmt} | t={dt:.2f}s"
            )

    summary = {
        "evaluator_model": model or EVALUATOR_MODEL,
        "n_sentences": len(overall_scores),
        "avg_overall": _avg(overall_scores),
        "avg_correctness": _avg(corr_scores),
        "avg_completeness": _avg(comp_scores),
        "avg_no_hallucination": _avg(nohall_scores),
        "avg_format_quality": _avg(fmt_scores),
        "out_jsonl": out_jsonl_path,
        "predictions_path": predictions_path,
    }

    print("\n==== RESUMEN JUEZ ====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("======================\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa sextetas con un LLM juez a partir de un JSON de predicciones."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Ruta al JSON con predicciones de sextetas (sentence -> list[sextet]).",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Ruta de salida del JSONL con las evaluaciones por frase.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Modelo del juez a usar (por defecto usa EVALUATOR_MODEL o el del entorno).",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Máx. nº de frases a evaluar (para pruebas rápidas).",
    )

    args = parser.parse_args()

    run_batch_eval(
        predictions_path=args.predictions,
        out_jsonl_path=args.out,
        model=args.model,
        max_sentences=args.max_sentences,
    )


if __name__ == "__main__":
    main()
