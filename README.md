# Airbnb Madrid — EDA y análisis exploratorio (LISTINGS)

## Descripción
Este proyecto realiza un análisis exploratorio (EDA) sobre un dataset de **Airbnb en Madrid** con el objetivo de:
- entender la estructura y calidad de los datos,
- detectar nulos y valores atípicos,
- preparar el dataset para análisis posteriores,
- explorar patrones de **precio**, **distribución territorial** y **disponibilidad**.

---

## Dataset
- **DataFrame:** `LISTINGS`
- **Tamaño:** `25.000 filas × 18 columnas`
- **Variables principales:**
  - Identificadores: `id`, `host_id`
  - Ubicación: `neighbourhood_group`, `neighbourhood`, `latitude`, `longitude`
  - Características del alojamiento: `room_type`, `minimum_nights`, `calculated_host_listings_count`
  - Actividad/demanda: `number_of_reviews`, `number_of_reviews_ltm`, `last_review`, `reviews_per_month`
  - Oferta: `availability_365`
  - Precio: `price`
  - Licencia: `license`

---

## Tipos de datos
- Numéricas: `id`, `host_id`, `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`,
  `calculated_host_listings_count`, `availability_365`, `number_of_reviews_ltm`
- Categóricas: `name`, `host_name`, `neighbourhood_group`, `neighbourhood`, `room_type`, `last_review`, `license`

---

## Calidad del dato

### Duplicados
- **Duplicados detectados:** `0`  
El dataset no contiene registros duplicados.

### Nulos (antes de limpieza)
Nulos relevantes detectados:
- `price`: **6.047 nulos (24,19%)**
- `last_review`: **5.147 nulos (20,59%)**
- `reviews_per_month`: **5.147 nulos (20,59%)**
- `license`: **15.812 nulos (63,25%)**
- `host_name`: **97 nulos**

---

## Limpieza aplicada
Se eliminaron columnas con bajo valor analítico para esta fase:

- `name` (texto libre con alta cardinalidad)
- `license` (gran porcentaje de nulos y formatos heterogéneos)
- `host_name` (información nominal sin impacto directo en el EDA)

---

## Transformación y tratamiento de nulos

### Creación de variables temporales
A partir de `last_review` se generaron:
- `last_rev_year`
- `last_rev_month`

### Identificación de alojamientos “inactivos”
Se detectaron **4.829 registros** con:
- `price` nulo **y**
- `availability_365 = 0`

Estos registros representan aproximadamente el **19,32%** del dataset total y se eliminaron por considerarse inactivos
(sesgan análisis de precio/demanda).

Tras la eliminación:
- `price` con nulos bajó a **6,04%**
- `last_rev_year`, `last_rev_month`, `reviews_per_month` con nulos pasaron a **16,99%**

### Imputación
- `price`: imputación con **mediana** (robusta ante outliers)
- `last_rev_year`: imputación con **0**
- `last_rev_month`: imputación con **0**
- `reviews_per_month`: imputación con **0**

Resultado:
- **Nulos finales en estas variables: 0**

---

## Análisis de precio

### Estadísticos principales (tras preparación)
- Media: **153,87€**
- Mediana: **110€**
- Min: **8€**
- Max: **25.654€**
- Desviación estándar alta: evidencia de **gran dispersión** y **outliers**.

**Conclusión:** el precio presenta una distribución muy asimétrica y los valores extremos elevan la media, por lo que
la **mediana** es una medida más representativa del precio típico.

### Precio por tipo de alojamiento (mediana)
| room_type          | mediana | IQR |
|-------------------|---------|-----|
| Entire home/apt    | 119,0   | 66  |
| Hotel room         | 145,0   | 59  |
| Private room       | 53,0    | 53  |
| Shared room        | 29,5    | 15  |

**Insights:**
- Los alojamientos completos y los hoteles son más caros.
- A mayor precio, mayor dispersión (IQR más alto) → segmentos más heterogéneos.
- Las habitaciones compartidas son más baratas y con precios más homogéneos.

### Diferencias por barrio
La comparación por barrio confirma una **heterogeneidad territorial**:
- Barrios periféricos (p.ej. Vinateros, Arcos, Campamento, Fontarrón, Los Ángeles) tienden a precios medianos más bajos.
- Barrios centrales y de mayor renta (p.ej. Castellana, Recoletos, Lista, Niño Jesús, Sol) concentran precios medianos más altos.

Además, se observa que barrios centrales como **Sol** y **Universidad** concentran valores extremos muy altos.

### Outliers
- Outliers detectados (regla empírica: media ± 3σ): **49**
- Se concentran en el extremo superior (precios extraordinariamente altos).
- Posibles causas:
  - alojamientos singulares o de lujo,
  - estrategias de pricing atípicas,
  - o errores de registro.

Para visualización y comparativas, se recomienda excluir outliers mediante **IQR** o usar transformaciones (p.ej. log).

### Relación “precio vs valoración” (reseñas por euro)
Los **Shared room** muestran la mejor eficiencia (más reseñas por euro), mientras que **Hotel room** es el peor caso.

---

## Disponibilidad y comportamiento de la oferta

### Distribución de disponibilidad anual (`availability_365`)
- Media: **210 días**
- Mediana: **243 días**
- Rango: **0 a 365 días**
- Alta dispersión: coexistencia de alojamientos muy activos y otros casi siempre disponibles.

### Barrios con mayor disponibilidad media (ejemplos)
Destacan zonas como:
- El Pardo, Atalaya, Horcajo, Valdemarín, Costillares, etc.

Estas áreas presentan una disponibilidad muy alta (cercana a 365), lo que podría sugerir:
- menor demanda,
- alojamientos de perfil menos turístico,
- o estrategias de disponibilidad permanente.

### Disponibilidad media por tipo de alojamiento
- Hotel room: **227,6**
- Private room: **216,8**
- Shared room: **215,3**
- Entire home/apt: **207,4**

Diferencias moderadas: los hoteles tienden a estar disponibles más días al año.

### Relación disponibilidad vs precio/demanda
Correlación Spearman:
- `availability_365` vs `price`: **0,0247** (relación prácticamente nula)
- `availability_365` vs `number_of_reviews_ltm`: **-0,1962** (relación negativa débil)

**Interpretación:**
- La disponibilidad no explica el precio.
- A mayor demanda (más reseñas recientes), menor disponibilidad, lo cual es coherente con ocupación más alta.

---

## Principales conclusiones
- El dataset está bien estructurado y sin duplicados, pero presenta nulos relevantes en variables clave (especialmente `price` y variables de reseñas).
- Existen outliers muy elevados en el precio que distorsionan la media; la mediana es más representativa.
- La localización influye fuertemente en el precio: centro más caro y periferia más barata.
- La disponibilidad se relaciona débilmente con la demanda (reseñas recientes), pero prácticamente nada con el precio.
- Tras eliminar alojamientos inactivos e imputar nulos, el dataset queda preparado para análisis más avanzados.

---

## Próximos pasos sugeridos
- Modelado de precio (regresión) incluyendo variables geográficas y tipo de alojamiento.
- Análisis por temporada/mes (si se incorporan datos temporales adicionales).
- Segmentación de alojamientos por perfil (turístico, larga estancia, alta disponibilidad, etc.).
- Validación de outliers: identificar si son lujo real o errores de datos.
