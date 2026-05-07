"""
Módulo de adquisición y preprocesamiento de audio para el proyecto LAPA.

Funcionalidades:
1. Descarga audio desde fuentes públicas autorizadas.
2. Convierte el audio a formato WAV.
3. Normaliza la señal.
4. Remuestrea el audio a 16 kHz mono.
5. Segmenta el audio en ventanas fijas para procesamiento posterior.

Aarón Madrigal Marín - C14373

ESTE CÓDIGO FUE GENERADO CON AYUDA DE HERRAMIENTAS DE INTELIGENCIA ARTIFICIAL, SE 
DEPURÓ TANTO MANUALMENTE COMO CON AYUDA DE LA HERRAMIENTA Y SE COMPROBÓ SU CORRECTO 
FUNCIONAMIENTO.


Ejemplo de ejecución:
python3 scripts/acquire_preprocess_audio.py 
  --url "https://www.youtube.com/watch?v=URL_DE_LA_SESION" 
  --segment_duration 30 
  --sample_rate 16000
o simplemente:
python3 scripts/acquire_preprocess_audio.py --url "https://www.youtube.com/watch?v=URL_DE_LA_SESION"

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


def segment_audio(
    input_path: str,
    output_dir: str,
    segment_duration: int = 30,
    sample_rate: int = 16000
) -> None:
    """
    Divide el audio en segmentos fijos.

    Args:
        input_path: Ruta del audio normalizado.
        output_dir: Carpeta donde se guardarán los segmentos.
        segment_duration: Duración de cada segmento en segundos.
        sample_rate: Frecuencia de muestreo del audio.
    """

    os.makedirs(output_dir, exist_ok=True)

	# Carga el audio procesado
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)

	# Cantidad de muestras por segmento
    samples_per_segment = segment_duration * sample_rate

    total_segments = int(np.ceil(len(audio) / samples_per_segment))

    base_name = os.path.splitext(os.path.basename(input_path))[0]

	# Recorre todos los segmentos
    for i in tqdm(range(total_segments), desc="Segmentando audio"):
        start = i * samples_per_segment
        end = start + samples_per_segment

        segment = audio[start:end]

		# Completa con ceros el último segmento
        if len(segment) < samples_per_segment:
            padding = samples_per_segment - len(segment)
            segment = np.pad(segment, (0, padding), mode="constant")

        segment_name = f"{base_name}_segment_{i:04d}.wav"
        segment_path = os.path.join(output_dir, segment_name)

		# Guarda el segmento
        sf.write(segment_path, segment, sample_rate)


def main():

	# Configuración de argumentos
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
    segment_audio(
        input_path=normalized_path,
        output_dir=args.segments_dir,
        segment_duration=args.segment_duration,
        sample_rate=args.sample_rate
    )

    print("Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()
