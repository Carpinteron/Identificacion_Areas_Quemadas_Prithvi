# Identificación de áreas quemadas con Prithvi

Proyecto del Grupo 1 del curso de Inteligencia Artificial. Segmenta cicatrices de incendio en imágenes satelitales HLS con el modelo fundacional **Prithvi** (NASA/IBM). El encoder va congelado y encima se entrena una cabeza ligera (Random Forest). El resultado se compara con una línea base y con controles.

- **Datos:** [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars): 6 bandas (azul, verde, rojo, NIR, SWIR1, SWIR2) a 30 m, con etiquetas a partir de perímetros MTBS. Se usa un recorte central de 224 × 224 píxeles.
- **Flujo:** imagen → parches → Prithvi congelado → Random Forest → máscara binaria (0 = no quemado, 1 = quemado, −1 = sin dato).
- **Requisitos de cómputo:** no hace falta GPU ni Colab Pro. Corre en Colab gratis y en un portátil sin GPU.

## Resultados principales

| Método (media por escena, 10 escenas de prueba) | F1 | IoU |
|---|---|---|
| Prithvi congelado + Random Forest | 0,64 | 0,55 |
| Línea base constante | 0,00 | 0,00 |

- La métrica principal es la **media por escena**. El agregado por píxel (F1 0,85) es solo un complemento, porque lo domina una escena grande que está casi toda quemada.
- Prithvi le gana a la línea base, pero con 20 escenas de entrenamiento **rinde igual que usar solo las bandas espectrales** con el mismo clasificador.
- **Con más imágenes de entrenamiento el modelo mejora, pero despacio:** con 60 escenas, el F1 por escena llega a 0,70 y la IoU a 0,58 (20 escenas de prueba). Prithvi queda por encima de las bandas solas, pero la diferencia no es concluyente.
- Predecir por bloques de 480 m pone un **techo de IoU de 0,81**.
- Los falsos negativos grandes vienen de la etiqueta: es el perímetro del incendio e incluye zonas que siguen verdes. Los falsos positivos son zonas más oscuras que su entorno, como sombras y vegetación rala.

## Contenido del repositorio

```
├── Grupo_1_quemadas_Prithvi.ipynb      Notebook principal (informe + código)
├── README.md
└── analisis_local/                     Análisis complementario ejecutado en local
    ├── analisis_20_imagenes.ipynb      Prueba ampliada a 20 escenas, controles, errores y curva de aprendizaje
    ├── curva_mas_imagenes.ipynb        Curva con 4, 10, 20, 40 y 60 escenas de entrenamiento (mismas 20 de prueba)
    ├── prithvi_taller.py               Módulo del flujo (copia del que escribe el notebook principal)
    ├── soporte_local.py                Descarga y selección de escenas para correr sin Colab
    ├── requirements.txt                Versiones exactas con las que se ejecutó (CPU)
    ├── VALORES_PARA_EL_NOTEBOOK.md     Cifras y tablas que alimentan las secciones del notebook principal
    ├── resultados/
    │   ├── figuras/                    Figuras numeradas (01 a 11) y una figura de resultado por escena
    │   ├── escenas/                    Las 20 escenas de prueba en .npz (etiqueta, predicción y línea base)
    │   ├── *.csv                       Métricas, matrices de confusión, controles, errores y curva de aprendizaje
    │   └── *.json                      Procedencia de los datos, normalización, casos y tiempos
    └── resultados_mas_imagenes/        Tablas y gráficas (figuras 12 y 13) de la curva con más imágenes
```

### Notebook principal: `Grupo_1_quemadas_Prithvi.ipynb`

Es el entregable. Contiene la portada, la declaración de contribuciones, la descripción del problema, el marco teórico, los datos y métodos, y el flujo experimental: descarga, exploración, normalización sin fuga de información, Prithvi congelado, cabeza Random Forest, evaluación frente a la línea base y exportación de evidencias. Después vienen el análisis de resultados, el análisis de errores, el análisis complementario, las preguntas de la ficha, la transferencia a Colombia, las consideraciones éticas, las conclusiones y las referencias.

### Análisis complementario: `analisis_local/`

Repite el flujo del notebook principal sin cambiar el modelo, la normalización ni las métricas, y lo extiende:

- **Reproducción** de la corrida de Colab (`reproduccion_vs_colab.csv`).
- **Prueba ampliada a 20 escenas**, con media por escena e intervalos de confianza (`metricas_por_escena.csv`, `metricas_resumen_20_escenas.csv`).
- **Controles:** la línea base, solo bandas espectrales con el mismo clasificador (`confusion_control_espectral.csv`) y el techo por bloques (`confusion_oraculo_parche.csv`).
- **Análisis de errores con cifras:** de dónde vienen los errores, cómo cambian con la distancia al borde, las firmas espectrales, la visibilidad de la cicatriz y el desempeño por región.
- **Curva de aprendizaje** según el número de imágenes de entrenamiento (`curva_aprendizaje.csv`).
- **Más imágenes de entrenamiento:** el mismo flujo con 4, 10, 20, 40 y 60 escenas, evaluado en las mismas 20 escenas de prueba, con gráficas de F1, IoU, precisión y recall según el número de imágenes (`curva_mas_imagenes.ipynb`, `resultados_mas_imagenes/`).
- **Línea base con índice NBR:** regla de umbral sobre el NBR (NIR y SWIR2), con el umbral elegido solo con entrenamiento, por píxel y por bloque de 16 × 16 (`metricas_nbr_por_escena.csv`, `metricas_comparacion_nbr.csv`, `umbral_nbr.json`; celdas nuevas en el notebook principal).
- **Extra Colombia:** contrato de entrada propuesto, etapas de validación y una celda que verifica el contrato sobre las escenas actuales (`verificacion_contrato_colombia.csv`).
- **Verificación** de que entrenamiento y prueba no comparten escenas (`independencia_train_test.csv`).

## Cómo ejecutar

**Notebook principal (Colab):** abrir `Grupo_1_quemadas_Prithvi.ipynb` en Google Colab y hacer *Entorno de ejecución → Reiniciar y ejecutar todo*. El notebook descarga los datos y los pesos de Prithvi.

**Análisis local (sin GPU):**

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r analisis_local/requirements.txt
jupyter notebook analisis_local/analisis_20_imagenes.ipynb
jupyter notebook analisis_local/curva_mas_imagenes.ipynb   # curva con 40 y 60 escenas: menos de 5 min
```

En CPU, cada imagen toma unos 0,45 s en el encoder. Las métricas pueden diferir de las de Colab con GPU en el tercer decimal.
