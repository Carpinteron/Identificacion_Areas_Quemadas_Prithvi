# Identificación de áreas quemadas con Prithvi
 
Proyecto del **Grupo 1** · Inteligencia Artificial · Prof. Eduardo Enrique Zurek Varela · Universidad del Norte · noviembre de 2026.
 
Segmenta cicatrices de incendio en imágenes satelitales HLS con el modelo fundacional **Prithvi-EO-1.0-100M** (IBM y NASA). El encoder va congelado y encima se entrena una cabeza ligera (Random Forest). El resultado se compara con una línea base constante y con controles.
 
- **Video (15 min):** [PENDIENTE: enlace]
- **Datos:** [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars): 6 bandas (azul, verde, rojo, NIR, SWIR1, SWIR2) a 30 m, con etiquetas a partir de perímetros de incendio MTBS. Se usa un recorte central de 224 × 224 píxeles.
- **Flujo:** imagen → parches de 16 × 16 → Prithvi congelado → Random Forest → máscara binaria (0 = no quemado, 1 = quemado, −1 = sin dato, excluido de las métricas).
- **Cómputo:** corre en Google Colab gratis. La ejecución final se hizo con GPU, y el análisis complementario también corre en un portátil sin GPU.
## Resultados principales
 
Corrida final en Colab: 20 escenas de entrenamiento y 10 de prueba. Clase quemada, **media por escena**.
 
| Método | F1 | IoU |
|---|---|---|
| Prithvi congelado + Random Forest | 0,64 | 0,55 |
| Bandas espectrales + el mismo Random Forest (control) | 0,64 | 0,53 |
| Regla de umbral sobre el NBR (umbral elegido con entrenamiento) | 0,56 | 0,49 |
| Línea base constante (siempre «no quemado») | 0,00 | 0,00 |
| Techo por bloques de 16 × 16 (usa etiquetas; no es un método) | 0,89 | 0,81 |
 
- La métrica principal es la **media por escena**. El agregado por píxel (F1 0,85, IoU 0,74) es un complemento: lo domina una escena que en el recorte está toda quemada. La línea base acierta el 76,9 % de los píxeles sin detectar ninguno quemado.
- Prithvi le gana a la línea base, pero con 20 escenas de entrenamiento **rinde igual que las bandas espectrales** con el mismo clasificador.
- **Con más imágenes el modelo mejora, pero despacio** (análisis local, 20 escenas de prueba): con 60 escenas de entrenamiento el F1 por escena llega a 0,70 y la IoU a 0,58. Prithvi queda por encima de las bandas solas, pero la diferencia no es concluyente.
- **Errores:** el 77 % está en bloques de borde, por predecir en bloques de 480 m, y el 72 % son falsos negativos. Rinde peor en el centro y el este de EE. UU. (F1 0,47 frente a 0,89 en el oeste), donde había menos escenas de entrenamiento. Los falsos positivos son zonas más oscuras que su entorno (sombras, vegetación rala). Varios falsos negativos grandes están en perímetros que en la imagen todavía se ven verdes.
## Contenido del repositorio
 
```
├── Grupo_1_quemadas_Prithvi.ipynb      Notebook principal (informe + código), ejecutado en Colab con salidas
├── README.md
├── resultados_colab/                   Evidencias de la ejecución final en Colab (ZIP exportado por el notebook)
└── analisis_local/                     Análisis complementario ejecutado en local
    ├── analisis_20_imagenes.ipynb      Prueba ampliada a 20 escenas, controles, errores y curva de aprendizaje
    ├── curva_mas_imagenes.ipynb        Curva con 4, 10, 20, 40 y 60 escenas de entrenamiento (mismas 20 de prueba)
    ├── prithvi_taller.py               Módulo del flujo (copia del que escribe el notebook principal)
    ├── soporte_local.py                Descarga y selección de escenas para correr sin Colab
    ├── requirements.txt                Versiones exactas con las que se ejecutó (CPU)
    ├── VALORES_PARA_EL_NOTEBOOK.md     Cifras y tablas del análisis local que se citan en el notebook principal
    ├── resultados/
    │   ├── figuras/                    Figuras numeradas y una figura de resultado por escena
    │   ├── escenas/                    Las 20 escenas de prueba en .npz (etiqueta, predicción y línea base)
    │   ├── *.csv                       Métricas, matrices de confusión, controles, errores y curva de aprendizaje
    │   └── *.json                      Procedencia de los datos, normalización, casos y tiempos
    └── resultados_mas_imagenes/        Tablas y gráficas de la curva con más imágenes
```
 
### Notebook principal: `Grupo_1_quemadas_Prithvi.ipynb`
 
Es el entregable. Contiene la portada, la declaración de contribuciones y de uso de IA, la descripción del problema, el marco teórico, los datos y métodos, y el flujo experimental: descarga, exploración, normalización sin fuga de información, Prithvi congelado, cabeza Random Forest, evaluación frente a la línea base, controles (bandas espectrales, techo por bloques y regla NBR) y exportación de evidencias. Después vienen el análisis de resultados, el análisis de errores, el análisis complementario (secciones A a H2), las preguntas de la ficha, la transferencia a Colombia, las consideraciones éticas, las conclusiones y las referencias en formato IEEE.
 
Las cifras sobre las 10 escenas de prueba salen de las salidas de este notebook. Las de 20 escenas de prueba y de 40 y 60 de entrenamiento salen de `analisis_local/` y en el texto están rotuladas como análisis local.
 
### Evidencias de Colab: `resultados_colab/`
 
Lo que exporta la celda «9. Exportar resultados y datos de prueba»: métricas por clase, por escena y media por escena, matrices de confusión, controles, umbral NBR, procedencia (modelo, revisión, versiones), normalización, manifiesto de datos con hashes, figuras y predicciones por escena.
 
### Análisis complementario: `analisis_local/`
 
Repite el flujo del notebook principal sin cambiar el modelo, la normalización ni las métricas, y lo extiende:
 
- **Reproducción** de la corrida de Colab (`reproduccion_vs_colab.csv`).
- **Prueba ampliada a 20 escenas**, con media por escena e intervalos de confianza (`metricas_por_escena.csv`, `metricas_resumen_20_escenas.csv`).
- **Controles:** la línea base, solo bandas espectrales con el mismo clasificador (`confusion_control_espectral.csv`) y el techo por bloques (`confusion_oraculo_parche.csv`).
- **Análisis de errores con cifras:** de dónde vienen los errores, cómo cambian con la distancia al borde, las firmas espectrales, la visibilidad de la cicatriz y el desempeño por región.
- **Curva de aprendizaje** según el número de imágenes de entrenamiento (`curva_aprendizaje.csv`).
- **Más imágenes de entrenamiento:** el mismo flujo con 4, 10, 20, 40 y 60 escenas, evaluado en las mismas 20 escenas de prueba (`curva_mas_imagenes.ipynb`, `resultados_mas_imagenes/`).
- **Línea base con índice NBR:** regla de umbral sobre el NBR, con el umbral elegido solo con entrenamiento, por píxel y por bloque de 16 × 16 (`metricas_nbr_por_escena.csv`, `metricas_comparacion_nbr.csv`, `umbral_nbr.json`).
- **Extra Colombia:** contrato de entrada propuesto, etapas de validación y verificación del contrato sobre las escenas actuales (`verificacion_contrato_colombia.csv`).
- **Verificación** de que entrenamiento y prueba no comparten escenas (`independencia_train_test.csv`).
## Cómo ejecutar
 
**Notebook principal (Colab):** abrir `Grupo_1_quemadas_Prithvi.ipynb` en Google Colab y hacer *Entorno de ejecución → Reiniciar y ejecutar todo*. El notebook descarga los datos (unos 1,9 GB leídos del archivo comprimido) y los pesos de Prithvi.
 
**Análisis local (sin GPU):**
 
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r analisis_local/requirements.txt
jupyter notebook analisis_local/analisis_20_imagenes.ipynb
jupyter notebook analisis_local/curva_mas_imagenes.ipynb   # curva con 40 y 60 escenas: menos de 5 min
```
 
En CPU, cada imagen toma unos 0,45 s en el encoder. Frente a Colab con GPU, las métricas agregadas difieren a lo sumo en 0,004; por escena las diferencias pueden llegar a 0,02 de F1.
 
## Integrantes
 
Wilmer Santiago · Isabella Arrieta Juliao · Paula Nuñez Zarante · Gabriel Andres · Natalia Carpintero Leal · Ivan Parra · Alejandra Restrepo · Alan Garcia · Andres Serrano · Bryan Montoya. Los aportes de cada uno están en la sección «Declaración de contribuciones y uso de ayudas» del notebook.
