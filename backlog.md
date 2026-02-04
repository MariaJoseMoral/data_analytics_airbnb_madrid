📌 BACKLOG DEL PROYECTO
Proyecto: Análisis de Datos de Airbnb en Madrid

🟦 ÉPICA 1 – Preparación del entorno
Objetivo:
  Disponer de toda la infraestructura técnica necesaria para comenzar el proyecto.
Tareas:
  Crear repositorio en GitHub para el proyecto.
  Definir estructura inicial del repositorio:
  Carpetas: data/, notebooks/, scripts/, docs/, sql/, visualizations/
  Crear archivo README.md con descripción del proyecto.
  Configurar entorno virtual de Python.
  Instalar librerías necesarias:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - sqlalchemy / sqlite / mysql
  Documentar requisitos en requirements.txt.

🟦 ÉPICA 2 – Obtención de datos
Objetivo:
  Descargar y preparar los datos necesarios desde la fuente InsideAirbnb.
Tareas:
  Identificar ciudad asignada para el análisis.
  Descargar datasets correspondientes:
    listings.csv
    reviews.csv (si aplica)
    calendar.csv (si aplica)
  Almacenar datasets originales en carpeta data/raw.
  Crear script de carga inicial de datos en Python.
  Verificar correcta lectura de archivos en pandas.

🟦 ÉPICA 3 – Análisis Exploratorio de Datos (EDA)
Objetivo:
Comprender la estructura y calidad de los datos.
Tareas:
Cargar dataset en un DataFrame.
Analizar número de filas y columnas.
Identificar número de alojamientos únicos.
Analizar tipos de alojamiento existentes.
Contar alojamientos por barrio/zona.
Analizar distribución geográfica.
Detectar valores nulos por columna.
Calcular porcentaje de nulos.
Identificar columnas irrelevantes.
Analizar tipos de datos.
Detectar valores atípicos en precios.
Analizar impacto de outliers en estadísticas.
Documentar conclusiones del EDA.
Entregable:
Notebook con análisis exploratorio comentado.
🟦 ÉPICA 4 – Transformación y limpieza de datos
Objetivo:
Garantizar un dataset limpio y listo para el análisis.
Tareas:
Convertir columnas numéricas mal tipadas (price, reviews_per_month, etc.).
Eliminar símbolos especiales de la columna precio.
Tratar valores nulos:
imputación cuando tenga sentido
eliminación cuando sea necesario
Detectar y eliminar duplicados.
Normalizar valores categóricos.
Corregir errores tipográficos.
Eliminar columnas sin valor analítico.
Crear funciones reutilizables de limpieza.
Guardar dataset limpio en data/processed.
Entregable:
Script o notebook con pipeline de limpieza.
🟦 ÉPICA 5 – Análisis y visualización
Objetivo:
Extraer insights mediante gráficos y métricas.
💰 Análisis de precios
Calcular precio medio y mediano.
Comparar precios por tipo de alojamiento.
Analizar precios por barrio.
Identificar barrios más caros y más baratos.
Relacionar precio con valoraciones.
Detectar precios extremadamente altos o bajos.
Crear gráficos:
histogramas
boxplots
mapas de calor
⭐ Análisis de valoraciones y reseñas
Calcular valoración media global.
Comparar valoraciones por tipo de alojamiento.
Analizar relación reseñas vs valoración.
Identificar barrios mejor valorados.
Analizar relación precio-valoración.
📅 Análisis de disponibilidad
Analizar distribución de disponibilidad anual.
Comparar disponibilidad por barrio.
Comparar disponibilidad por tipo de alojamiento.
Relacionar disponibilidad con precio.
Crear visualizaciones específicas de disponibilidad.
Generación de informe visual
Unificar todos los gráficos en un notebook final.
Redactar conclusiones analíticas.
Preparar storytelling de datos.
🟦 ÉPICA 6 – Diseño de Base de Datos (BONUS)
Objetivo:
Estructurar los datos en un modelo relacional.
Tareas:
Diseñar modelo conceptual de BD.
Definir tablas principales:
alojamientos
anfitriones
localización
reseñas
disponibilidad
Definir claves primarias.
Definir claves foráneas.
Crear diagramas entidad-relación.
Implementar base de datos en SQL.
Crear scripts de creación de tablas.
Insertar datos limpios en la base de datos.
Automatizar inserciones con Python.
🟦 ÉPICA 7 – Automatización
Objetivo:
Que todo el flujo sea reproducible.
Tareas:
Crear funciones para:
carga de datos
limpieza
inserción en BD
Unificar todo en un script principal.
Probar ejecución completa end-to-end.
🟦 ÉPICA 8 – Documentación y entrega
Objetivo:
Preparar la entrega final del proyecto.
Tareas:
Redactar memoria del proyecto:
objetivos
metodología
resultados
conclusiones
Documentar uso del repositorio.
Subir todo el código a GitHub.
Preparar presentación final.
Ensayar explicación de resultados.
📦 ENTREGABLES FINALES
Repositorio GitHub organizado
Notebooks de EDA y análisis
Scripts de limpieza y automatización
Base de datos funcional (bonus)
Informe final con visualizaciones y conclusiones
