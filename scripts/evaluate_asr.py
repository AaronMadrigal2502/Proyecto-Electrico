"""
Módulo de evaluación cuantitativa para transcripciones ASR del proyecto LAPA.

Funcionalidades:
1. Lee una transcripción de referencia escrita manualmente.
2. Lee una transcripción generada automáticamente por Whisper.
3. Calcula WER (Word Error Rate).
4. Calcula CER (Character Error Rate).
5. Guarda los resultados en un archivo CSV.

Ejemplo de ejecución:
python3 scripts/evaluate_asr.py \
  --reference data/references/sesion_001.txt \
  --hypothesis data/transcripts/sesion_001_transcription.csv \
  --output data/evaluation/sesion_001_metrics.csv
"""

import os
import csv
import argparse
import unicodedata
from typing import Dict, List

from jiwer import wer, cer, process_words


def read_txt(path: str) -> str:
    """
    Lee un archivo de texto plano.
    """
    with open(path, mode="r", encoding="utf-8") as file:
        return file.read().strip()


def read_transcript_csv(path: str, text_column: str = "transcripcion") -> str:
    """
    Lee un CSV de transcripción y concatena el texto de sus segmentos.
    """
    texts: List[str] = []

    with open(path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if text_column not in reader.fieldnames:
            raise ValueError(
                f"No se encontró la columna '{text_column}' en {path}. "
                f"Columnas disponibles: {reader.fieldnames}"
            )

        for row in reader:
            text = row.get(text_column, "").strip()

            if text:
                texts.append(text)

    return " ".join(texts)


def load_text(path: str, text_column: str = "transcripcion") -> str:
    """
    Carga texto desde .txt o desde .csv.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == ".txt":
        return read_txt(path)

    if extension == ".csv":
        return read_transcript_csv(path, text_column=text_column)

    raise ValueError("Formato no soportado. Use archivos .txt o .csv.")


def normalize_text(text: str, keep_accents: bool = True) -> str:
    """
    Normaliza texto para una comparación más consistente.
    """
    text = text.lower()
    text = text.replace("\n", " ")

    if not keep_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    punctuation = ".,;:¿?¡!\"'()[]{}«»“”‘’—–-"

    for mark in punctuation:
        text = text.replace(mark, " ")

    text = " ".join(text.split())

    return text


def evaluate_asr(
    reference_text: str,
    hypothesis_text: str,
    normalize: bool = True,
    keep_accents: bool = True
) -> Dict[str, float]:
    """
    Calcula WER, CER y conteos de errores.
    """
    if normalize:
        reference_text = normalize_text(reference_text, keep_accents=keep_accents)
        hypothesis_text = normalize_text(hypothesis_text, keep_accents=keep_accents)

    word_metrics = process_words(reference_text, hypothesis_text)

    return {
        "wer": wer(reference_text, hypothesis_text),
        "cer": cer(reference_text, hypothesis_text),
        "hits": word_metrics.hits,
        "substitutions": word_metrics.substitutions,
        "deletions": word_metrics.deletions,
        "insertions": word_metrics.insertions,
        "reference_words": len(reference_text.split()),
        "hypothesis_words": len(hypothesis_text.split()),
    }


def save_results(output_path: str, session_name: str, metrics: Dict[str, float]) -> None:
    """
    Guarda las métricas en un archivo CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "sesion",
        "wer",
        "cer",
        "hits",
        "substitutions",
        "deletions",
        "insertions",
        "reference_words",
        "hypothesis_words",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        row = {"sesion": session_name}
        row.update(metrics)

        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa transcripciones ASR usando WER y CER."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Ruta del texto de referencia manual (.txt)."
    )

    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Ruta de la transcripción generada por Whisper (.txt o .csv)."
    )

    parser.add_argument(
        "--output",
        default="data/evaluation/asr_metrics.csv",
        help="Ruta del CSV donde se guardarán las métricas."
    )

    parser.add_argument(
        "--text_column",
        default="transcripcion",
        help="Nombre de la columna de texto si la hipótesis es un CSV."
    )

    parser.add_argument(
        "--session_name",
        default=None,
        help="Nombre de la sesión evaluada. Si se omite, se usa el nombre del archivo."
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Desactiva la normalización básica de texto."
    )

    parser.add_argument(
        "--ignore_accents",
        action="store_true",
        help="Ignora diferencias por tildes durante la evaluación."
    )

    args = parser.parse_args()

    reference_text = load_text(args.reference)
    hypothesis_text = load_text(args.hypothesis, text_column=args.text_column)

    session_name = args.session_name

    if session_name is None:
        session_name = os.path.splitext(os.path.basename(args.hypothesis))[0]

    metrics = evaluate_asr(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        normalize=not args.no_normalize,
        keep_accents=not args.ignore_accents
    )

    save_results(args.output, session_name, metrics)

    print("Evaluación finalizada.")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
