# Identificación de áreas quemadas con Prithvi

Proyecto del Grupo 1 (curso de IA) para segmentar áreas quemadas en imágenes satelitales con el modelo fundacional **Prithvi** (NASA/IBM).

## Contenido

- `Grupo_1_quemadas_Prithvi.ipynb`: notebook principal.

## Flujo del notebook

1. Descarga de imágenes y etiquetas reales.
2. Lectura, verificación y exploración de los datos.
3. Normalización sin fuga de información del conjunto de prueba.
4. Descarga de Prithvi y congelamiento del encoder.
5. Entrenamiento exclusivo de una cabeza ligera de segmentación.
6. Evaluación frente a una línea base.
7. Exportación de resultados y datos de prueba.

## Uso

Abre el notebook en Jupyter o Google Colab y ejecuta las celdas en orden.
