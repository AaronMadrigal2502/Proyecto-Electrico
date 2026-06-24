# Módulo de adquisición y preprocesamiento de audio - LAPA

## Descripción

Este módulo permite automatizar la descarga y preparación inicial de sesiones legislativas provenientes de fuentes públicas autorizadas, como el canal oficial de la Asamblea Legislativa de Costa Rica.

El objetivo es generar archivos de audio normalizados y segmentados para su uso posterior en tareas de reconocimiento automático del habla, análisis acústico o procesamiento lingüístico.

## Funcionalidades

- Descarga de audio desde una URL pública.
- Conversión automática a formato WAV.
- Remuestreo a 16 kHz.
- Conversión a canal mono.
- Normalización básica de amplitud.
- Segmentación automática en ventanas dictadas por pausas de sensibilidad ajustable.

## Dependencias

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

Este proyecto requiere FFmpeg instalado en el sistema.

Ubuntu:
```bash
sudo apt install ffmpeg
```

Se recomienda utilizar `pip3` para asegurar que las librerías se instalen en el entorno de Python 3:

```bash
pip3 install -r requirements.txt
```

Además, para evitar errores de descarga desde YouTube, se recomienda mantener actualizada la herramienta yt-dlp:
```
pip3 install -U yt-dlp
```
## Ejecución
Ejemplo:
```
python3 scripts/acquire_preprocess_audio.py \
  --url "URL_DE_LA_SESION" \
  --top_db 30 \
  --min_segment_duration 2 \
  --merge_gap 2 \
  --padding 0.3 \
  --whisper_model base
```

```
python3 scripts/evaluate_asr.py \
  --reference data/references/ejemplo1.txt \
  --hypothesis "data/transcripts/ejemplo1_transcription.csv" \
  --output data/evaluation/ejemplo1_metrics.csv
```
Se pueden ignorar tildes durante la evaluación agregando ```--ignore_accents``` al final.
