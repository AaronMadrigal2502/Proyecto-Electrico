"""
Módulo de adquisición y preprocesamiento de audio para el proyecto LAPA.

Funcionalidades:
1. Descarga audio desde fuentes públicas autorizadas.
2. Convierte el audio a formato WAV.
3. Normaliza la señal.
4. Remuestrea el audio a 16 kHz mono.
5. Segmenta automáticamente el audio según regiones con actividad de voz.
6. Transcribe automáticamente los segmentos usando Whisper.

Aarón Madrigal Marín - C14373

ESTE CÓDIGO FUE GENERADO CON AYUDA DE HERRAMIENTAS DE INTELIGENCIA ARTIFICIAL, SE 
DEPURÓ TANTO MANUALMENTE COMO CON AYUDA DE LA HERRAMIENTA Y SE COMPROBÓ SU CORRECTO 
FUNCIONAMIENTO.


Ejemplo de ejecución:
python3 scripts/acquire_preprocess_audio.py \
  --url "https://www.youtube.com/watch?v=URL_DE_LA_SESION" \
  --top_db 30 \
  --min_segment_duration 1.0 \
  --sample_rate 16000 \
  --whisper_model small \
  --language es  

"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import csv
import whisper
from yt_dlp import YoutubeDL
from tqdm import tqdm


def download_audio(url: str, output_dir: str) -> str:
    """
    Descarga el audio desde una URL pública.

    Args:
        url: Enlace.
        output_dir: Carpeta donde se guardará el archivo descargado.

    Returns:
        Ruta del archivo WAV descargado.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {

		# Prioriza audio m4a por estabilidad
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,

	# Reintentos en caso de fallos
	"continuedl": True,
	"retries": 10,
	"fragment_retries": 10,
	"ignoreerrors": False,

		# Conversión a .wav
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

	# Fuerza el audio a mono y 16kHz
	"postprocessor_args": [
		"-ar", "16000",
		"-ac", "1"
	],
        "quiet": False,
    }

	# Descarga el audio
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return os.path.splitext(filename)[0] + ".wav"


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normaliza la amplitud del audio.

    Args:
        audio: Señal de audio.

    Returns:
        Señal normalizada.
    """

    max_amplitude = np.max(np.abs(audio))

    if max_amplitude == 0:
        return audio

    return audio / max_amplitude


def preprocess_audio(input_path: str, output_path: str, sample_rate: int = 16000) -> None:
    """
    Convierte el audio a mono, 16 kHz y lo normaliza.

    Args:
        input_path: Ruta del audio original.
        output_path: Ruta del audio normalizado.
        sample_rate: Frecuencia de muestreo objetivo.
    """

	# Carga y remuestrea
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)

	# Normaliza la señal
    audio = normalize_audio(audio)

	# Crea la carpeta de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

	# Guarda el audio procesado
    sf.write(output_path, audio, sample_rate)

def merge_close_intervals(intervals, sample_rate, merge_gap: float = 2.0):
    """
    Une segmentos separados por pausas cortas.
    """

    if len(intervals) == 0:
        return intervals

    max_gap_samples = int(merge_gap * sample_rate)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]

        if start - previous_end <= max_gap_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return np.array(merged)

def segment_audio_by_voice_activity(
    input_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    top_db: int = 30,
    min_segment_duration: float = 1.0,
    merge_gap: float = 2.0,
    padding: float = 0.3
) -> str:

    """
    Segmenta el audio usando detección de actividad de voz.
    """

    os.makedirs(output_dir, exist_ok=True)

    audio, sr = librosa.load(
        input_path,
        sr=sample_rate,
        mono=True
    )

    intervals = librosa.effects.split(
        audio,
        top_db=top_db
    )

	intervals = merge_close_intervals(
    intervals,
    sample_rate,
    merge_gap=merge_gap
    )

    base_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    metadata_path = os.path.join(
        output_dir,
        f"{base_name}_metadata.csv"
    )

    with open(
        metadata_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "segmento",
            "inicio",
            "fin",
            "duracion",
            "archivo"
        ])

        segment_index = 0

        for start_sample, end_sample in intervals:

            duration = (
                end_sample - start_sample
            ) / sample_rate

            # Ignora segmentos muy cortos
            if duration < min_segment_duration:
                continue

            padding_samples = int(padding * sample_rate)

            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(audio), end_sample + padding_samples)

            segment = audio[start_sample:end_sample]

            segment_name = (
                f"{base_name}_segment_"
                f"{segment_index:04d}.wav"
            )

            segment_path = os.path.join(
                output_dir,
                segment_name
            )

            sf.write(
                segment_path,
                segment,
                sample_rate
            )

            writer.writerow([
                segment_index,
                round(start_sample / sample_rate, 3),
                round(end_sample / sample_rate, 3),
                round(duration, 3),
                segment_name
            ])

            segment_index += 1
    return metadata_path

def transcribe_segments(
    metadata_path: str,
    segments_dir: str,
    output_path: str,
    model_name: str = "small",
    language: str = "es"
) -> None:
    """
    Transcribe los segmentos generados usando Whisper.

    Args:
        metadata_path: CSV generado durante la segmentación.
        segments_dir: Carpeta donde están los segmentos WAV.
        output_path: Ruta del CSV final con transcripciones.
        model_name: Modelo de Whisper a utilizar.
        language: Idioma esperado del audio.
    """

    model = whisper.load_model(model_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(metadata_path, mode="r", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        rows = list(reader)

    with open(output_path, mode="w", newline="", encoding="utf-8") as output_file:
        fieldnames = ["segmento", "inicio", "fin", "duracion", "archivo", "transcripcion"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="Transcribiendo segmentos"):
            segment_path = os.path.join(segments_dir, row["archivo"])
			
			result = model.transcribe(
                segment_path,
                language=language,
                task="transcribe",
                fp16=False,
                no_speech_threshold=0.95,
                logprob_threshold=-2.0,
                condition_on_previous_text=False,
                initial_prompt=(
                    "Transcripción de una sesión legislativa de Costa Rica. "
                    "Se mencionan diputadas, diputados, Asamblea Legislativa, "
                    "comisiones legislativas, proyectos de ley, expedientes, mociones, "
                    "Presidencia, Plenario Legislativo y votaciones."
                )
            )

            text = result.get("text", "").strip()

            if text == "":
                print(f"Advertencia: Whisper no generó texto para {segment_path}")

            row["transcripcion"] = text
            writer.writerow(row)



def main():

	# Configuración de argumentos
    parser = argparse.ArgumentParser(
        description="Descarga, normaliza y segmenta sesiones legislativas para el proyecto LAPA."
    )

    parser.add_argument("--url", required=True, help="URL pública de la sesión legislativa.")
    parser.add_argument("--raw_dir", default="data/raw", help="Carpeta para audio original.")
    parser.add_argument("--normalized_dir", default="data/normalized", help="Carpeta para audio normalizado.")
    parser.add_argument("--segments_dir", default="data/segments", help="Carpeta para segmentos.")
    parser.add_argument("--transcripts_dir", default="data/transcripts", help="Carpeta para guardar las transcripciones.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Frecuencia de muestreo objetivo.")
    parser.add_argument("--top_db", type=int, default=30, help="Sensibilidad para detección de voz.")
    parser.add_argument("--min_segment_duration", type=float, default=1.0, help="Duración mínima de un segmento.")
    parser.add_argument("--whisper_model", default="small", help="Modelo de Whisper: tiny, base, small, medium o large.")
    parser.add_argument("--language", default="es", help="Idioma del audio para Whisper.")
    parser.add_argument("--skip_transcription", action="store_true", help="Solo segmenta el audio, sin transcribir.")
    parser.add_argument("--merge_gap", type=float, default=2.0, help="Une segmentos separados por pausas menores a este valor en segundos.")
    parser.add_argument("--padding", type=float, default=0.3, help="Margen agregado antes y después de cada segmento.")
	
    args = parser.parse_args()

    print("Descargando audio...")

	# Se descarga el audio original
    raw_audio_path = download_audio(args.url, args.raw_dir)

    normalized_path = os.path.join(
        args.normalized_dir,
        os.path.basename(raw_audio_path)
    )

    print("Preprocesando audio...")

	# Se procesa el audio
    preprocess_audio(
        input_path=raw_audio_path,
        output_path=normalized_path,
        sample_rate=args.sample_rate
    )

    print("Segmentando audio...")
	
	# Se generan los segmentos
    metadata_path = segment_audio_by_voice_activity(
        input_path=normalized_path,
        output_dir=args.segments_dir,
        sample_rate=args.sample_rate,
        top_db=args.top_db,
        min_segment_duration=args.min_segment_duration,
        merge_gap=args.merge_gap,
        padding=args.padding
    )

    if not args.skip_transcription:
        base_name = os.path.splitext(os.path.basename(normalized_path))[0]
        transcript_path = os.path.join(
            args.transcripts_dir,
            f"{base_name}_transcription.csv"
        )

        print("Transcribiendo segmentos con Whisper...")
        
        transcribe_segments(
            metadata_path=metadata_path,
            segments_dir=args.segments_dir,
            output_path=transcript_path,
            model_name=args.whisper_model,
            language=args.language
        )

        print(f"Transcripción guardada en: {transcript_path}")

    print("Proceso finalizado correctamente.")

if __name__ == "__main__":
    main()
