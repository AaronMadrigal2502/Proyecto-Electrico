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
- Segmentación automática en ventanas de duración fija.

## Dependencias

Instalar las dependencias con:

```bash
pip install -r requirements.txt

Este proyecto requiere FFmpeg instalado en el sistema.

Ubuntu:
sudo apt install ffmpeg
