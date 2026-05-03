"""
Módulo de adquisición y preprocesamiento de audio para el proyecto LAPA.

Funcionalidades:
1. Descarga audio desde fuentes públicas autorizadas.
2. Convierte el audio a formato WAV.
3. Normaliza la señal.
4. Remuestrea el audio a 16 kHz mono.
5. Segmenta el audio en ventanas fijas para procesamiento posterior.

Aarón Madrigal Marín - C14373

Ejemplo de ejecución:
python scripts/acquire_preprocess_audio.py \
  --url "https://www.youtube.com/watch?v=URL_DE_LA_SESION" \
  --segment_duration 30 \
  --sample_rate 16000

"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from yt_dlp import YoutubeDL
from tqdm import tqdm


def download_audio(url: str, output_dir: str) -> str:
    """
    Descarga el audio de una sesión legislativa desde una URL pública.

    Args:
        url: Enlace de la sesión legislativa.
        output_dir: Carpeta donde se guardará el archivo descargado.

    Returns:
        Ruta del archivo WAV descargado.
    """

    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
	"continuedl": True,
	"retries": 10,
	"fragment_retries": 10,
	"ignoreerrors": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
	"postprocessor_args": [
		"-ar", "16000",
		"-ac", "1"
	],
        "quiet": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return os.path.splitext(filename)[0] + ".wav"


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normaliza la amplitud del audio al rango [-1, 1].

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
    Carga, remuestrea, convierte a mono y normaliza un archivo de audio.

    Args:
        input_path: Ruta del audio original.
        output_path: Ruta del audio normalizado.
        sample_rate: Frecuencia de muestreo objetivo.
    """

    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    audio = normalize_audio(audio)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, sample_rate)


def segment_audio(
    input_path: str,
    output_dir: str,
    segment_duration: int = 30,
    sample_rate: int = 16000
) -> None:
    """
    Segmenta un archivo de audio en ventanas de duración fija.

    Args:
        input_path: Ruta del audio normalizado.
        output_dir: Carpeta donde se guardarán los segmentos.
        segment_duration: Duración de cada segmento en segundos.
        sample_rate: Frecuencia de muestreo del audio.
    """

    os.makedirs(output_dir, exist_ok=True)

    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    samples_per_segment = segment_duration * sample_rate

    total_segments = int(np.ceil(len(audio) / samples_per_segment))

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i in tqdm(range(total_segments), desc="Segmentando audio"):
        start = i * samples_per_segment
        end = start + samples_per_segment

        segment = audio[start:end]

        if len(segment) < samples_per_segment:
            padding = samples_per_segment - len(segment)
            segment = np.pad(segment, (0, padding), mode="constant")

        segment_name = f"{base_name}_segment_{i:04d}.wav"
        segment_path = os.path.join(output_dir, segment_name)

        sf.write(segment_path, segment, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Descarga, normaliza y segmenta sesiones legislativas para el proyecto LAPA."
    )

    parser.add_argument("--url", required=True, help="URL pública de la sesión legislativa.")
    parser.add_argument("--raw_dir", default="data/raw", help="Carpeta para audio original.")
    parser.add_argument("--normalized_dir", default="data/normalized", help="Carpeta para audio normalizado.")
    parser.add_argument("--segments_dir", default="data/segments", help="Carpeta para segmentos.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Frecuencia de muestreo objetivo.")
    parser.add_argument("--segment_duration", type=int, default=30, help="Duración de segmentos en segundos.")

    args = parser.parse_args()

    print("Descargando audio...")
    raw_audio_path = download_audio(args.url, args.raw_dir)

    normalized_path = os.path.join(
        args.normalized_dir,
        os.path.basename(raw_audio_path)
    )

    print("Preprocesando audio...")
    preprocess_audio(
        input_path=raw_audio_path,
        output_path=normalized_path,
        sample_rate=args.sample_rate
    )

    print("Segmentando audio...")
    segment_audio(
        input_path=normalized_path,
        output_dir=args.segments_dir,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate
    )

    print("Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()
