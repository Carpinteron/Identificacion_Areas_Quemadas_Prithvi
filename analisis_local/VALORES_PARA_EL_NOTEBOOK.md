# Valores para completar los [PENDIENTE] del notebook principal

**Origen:** ejecución local en CPU (22-nov-2026) de `analisis_20_imagenes.ipynb` y una prueba de integración: el notebook principal actual se ejecutó completo con sus celdas nuevas (32 y 34), con los mismos datos que en Colab. En Colab con GPU, las métricas de Prithvi pueden variar hasta ±0.004. El control espectral y el techo por parche no dependen de la GPU.

**No hace falta GPU ni Colab Pro.** En un portátil i5 sin GPU, cada imagen toma ≈0.45 s en el encoder. Las celdas 32 y 34 corren en segundos.

---

## Análisis de resultados (P3)

**Media por escena en las 10 escenas de prueba de Colab** (la convención que pide el enunciado):

| Método | F1 | IoU | Precisión | Recall | Escenas |
|---|---|---|---|---|---|
| Prithvi congelado + Random Forest | 0.642 ± 0.352 | 0.551 ± 0.339 | 0.737 ± 0.356 | 0.610 ± 0.373 | 10 |
| Línea base constante | 0.000 | 0.000 | 0.000 | 0.000 | 10 |

± = desviación estándar entre escenas. IC95 % por bootstrap de escenas: F1 0.42–0.83 e IoU 0.35–0.74.

La tabla actual del notebook (F1 0.850, IoU 0.739) es el **agregado por píxel**. La diferencia con la media por escena viene sobre todo de la escena 0 (T10TGK): está 100 % quemada en el recorte, tiene 50 176 píxeles y F1 0.99, así que domina el agregado. Sugerencia: poner la media por escena como métrica principal y dejar el agregado rotulado como complemento.

**F1 por escena (Prithvi + RF):**

| Escena | 0 T10TGK | 1 T10TGL | 2 T11TQG | 3 T12SUJ | 4 T14RNV | 5 T14SKF | 6 T15TVN | 7 T16REV | 8 T16SDC | 9 T18SVJ |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.990 | 0.819 | 0.896 | 0.863 | 0.884 | 0.678 | 0.000 | 0.109 | 0.377 | 0.801 |

## Controles (P5)

| Método | F1 media por escena | IoU media por escena | F1 agregado | IoU agregado |
|---|---|---|---|---|
| Prithvi congelado + RF | 0.642 | 0.551 | 0.851 | 0.741 |
| Bandas espectrales + el mismo RF (celda 34) | 0.644 | 0.534 | 0.838 | 0.721 |
| Techo por parche 16×16 (usa etiquetas; no es un método) | 0.890 | 0.805 | 0.936 | 0.879 |

- Prithvi supera al control espectral en 5 de las 10 escenas (por F1).
- La celda 34 hoy muestra solo el agregado. La media por escena sale de `metricas_controles_por_escena.csv` y es la que pide el enunciado.
- Con 20 escenas de prueba y un control de 15 rasgos (análisis local), el resultado es el mismo: IoU 0.524 para Prithvi y 0.543 para el control; diferencia −0.019, IC95 −0.062 a +0.029.

Frases sugeridas: *En este subconjunto, las representaciones de Prithvi congelado no superan a un Random Forest con estadísticas espectrales simples. La cicatriz tiene una firma espectral fuerte (NBR) que un clasificador sencillo aprovecha sin contexto aprendido. El techo por parche muestra que, aun acertando todos los bloques, la predicción por bloques de 480 m limita la IoU a 0.81 en promedio.*

## Análisis de errores (P4)

- **Caso 1, acierto, escena 2 (T11TQG):** F1 = 0.896.
- **Caso 2, falso negativo, escena 6 (T15TVN):** F1 = 0.000. En la escena completa, el NBR mediano dentro del perímetro es +0.18 y en el fondo +0.25: la cicatriz casi no se distingue en NBR, lo que es compatible con rebrote de primavera o con una quema de pastizal o humedal.
- **Caso 3, falso positivo, escena 1 (T10TGL):** F1 = 0.819, con 3 058 píxeles FP. Lo marcado por error es más oscuro que el fondo bien clasificado de la misma escena (brillo visible −0.017, NIR −0.035), con un NBR casi igual (−0.008). Es la firma de una sombra de ladera, y confirma lo que dice el borrador.
- **Escena 0 (T10TGK):** F1 = 0.990.
- **Patrón general medido** (20 escenas de prueba del análisis local):
  - En las 15 escenas comparables, lo que el modelo no detecta tiene NBR mayor que lo que sí detecta, y NDVI mayor en 14. El modelo siempre deja sin marcar la parte más verde del perímetro.
  - En 12 de 13 escenas, los falsos positivos tienen NIR menor que el fondo bien clasificado: zonas más oscuras, como sombras o vegetación rala.
  - El 46 % de los píxeles mal clasificados está en parches de borde; el 60 % de los FP está a ≤ 240 m del borde.

## Corrida con más imágenes (hecha: `curva_mas_imagenes.ipynb`)

- Mismo flujo con 4, 10, 20, 40 y 60 escenas de entrenamiento, evaluado siempre en las mismas 20 escenas de prueba. Sin GPU: menos de 5 min en el portátil.
- Media por escena de Prithvi + RF (F1 / IoU): 4 → 0.425 / 0.316; 10 → 0.587 / 0.470; 20 → 0.633 / 0.524; 40 → 0.663 / 0.543; 60 → 0.703 / 0.578.
- Control espectral (IoU): 0.454, 0.483, 0.543, 0.523 y 0.531. Con 60 escenas, Prithvi − control = +0.047 de IoU (IC95 de −0.013 a +0.109): Prithvi queda adelante, pero la diferencia no es concluyente.
- De 20 a 60 escenas sube sobre todo el recall (de 0.60 a 0.70). La mejora se concentra en las 5 escenas difíciles (F1 de 0.15 a 0.43) y en el centro y el este (de 0.48 a 0.61).
- Ya está en el notebook principal: tabla y viñetas en "Análisis de resultados", gráficas en la sección H2 y conclusiones.

## Preguntas de la ficha (P8): evidencia que se puede citar

- **P1, región oscura:** los falsos positivos son más oscuros que su entorno (NIR menor en 12 de 13 escenas); por ejemplo, las laderas en sombra de T10TGL.
- **P2, NIR y SWIR:** lo detectado como quemado tiene NBR mediano −0.25; el fondo bien clasificado, +0.13.
- **P3, tiempo desde el incendio:** en la prueba ampliada, la escena T11SLB.2018197 (16-jul-2018) está a 4.5 km de las coordenadas del Lions Fire (Ansel Adams Wilderness). Ese incendio empezó el 11-jun-2018 y siguió creciendo hasta agosto (13 347 acres al final; inactivo desde el 1-oct-2018). El área etiquetada (57.7 km²) coincide con el tamaño final del incendio (54.0 km²), pero solo el 18 % de esa área tenía firma de quemado en la fecha de la imagen. Es un ejemplo medible de que la etiqueta y la imagen pueden no corresponder al mismo momento. Fuentes: Wikipedia "Lions Fire" (con base en InciWeb) y los mapas del incidente CA-SNF-000493.

## Conclusiones (P1)

- Agregar la media por escena (F1 0.64) junto al agregado (0.85).
- Matizar "las representaciones de Prithvi sirven": superan a la línea base, pero en este subconjunto no a un control con bandas espectrales simples.
- Resultado de la corrida con más imágenes: ya está en Conclusiones (F1 0.70 e IoU 0.58 con 60 escenas).

## Ejecución (P1)

- Las celdas 32 y 34 todavía no tienen salidas en el notebook: hay que hacer "Reiniciar y ejecutar todo" en Colab y guardar con salidas. En la prueba local corrieron sin errores.
- Si se quiere incluir el análisis local completo, se pega la celda portátil y las secciones 5 a 11 de `analisis_20_imagenes.ipynb` después de la celda 34. Esa combinación se probó y corre de principio a fin.
