
## Diseño del Conjunto de Datos
Diseña y estructura tu conjunto de datos para obtener resultados óptimos en el entrenamiento del modelo.

## Preprocesamiento de Texto
- Limpiar datos de texto
- Eliminar palabras vacías (stopwords)
- Aplicar vectorización TF-IDF
- Normalizar y tokenizar contenido
- Manejo de caracteres especiales y acentos
- División en frases y párrafos

## Configuración del Entorno
Instala Docker y Unsloth para entrenamiento acelerado con GPU:
```bash
docker run --gpus all -it --rm -v "${PWD}:/app" -w /app unsloth/unsloth:latest
```

```bash
docker exec -it $(docker ps -q) bash
```

## Ejecución en GPU
Ejecuta el entrenamiento con soporte de GPU utilizando el contenedor Docker anterior. Monitorea el uso de memoria y optimiza los recursos disponibles.

## Ajuste de Parámetros
Evalúa y optimiza los hiperparámetros del modelo:
- Tasa de aprendizaje
- Tamaño del lote
- Número de épocas
- Pasos de acumulación de gradientes
- Regularización L1 y L2

## Ajuste del Fine Tunning
Implementa mejoras iterativas basadas en métricas de evaluación y resultados de rendimiento. Realiza validación cruzada y pruebas exhaustivas.