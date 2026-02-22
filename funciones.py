#%%
# Cargando librerías

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# Librerías de visualización
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
# Configuración
# -----------------------------------------------------------------------

import math
#%%
# Cargando el dataset
df_listings = pd.read_csv('listings.csv')
df_reviews = pd.read_csv('reviews.csv')
df = pd.read_csv('listings_eda.csv', index_col= 0)

pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames

#%% 
# Cargando función: EDA 
def eda(df, name):
    
    print(f"\n{'='*40}")
    print(f"EDA DEL DATAFRAME: {name.upper()}")
    print(f"{'='*40}\n")
    
    print("========== RESUMEN GENERAL ==========")
    print(f"Filas x Columnas (shape): {df.shape}")
    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nNulos por columna:")
    print(df.isnull().sum())

    print("\n========== DESCRIBE NUMÉRICO ==========")
    print(df.describe().T)

    print("\n========== DESCRIBE CATEGÓRICO (object/category) ==========")
    print(df.describe(include=["O"]).T)

    print("\n========== HEAD ==========")
    print(df.head())
    print("\n========== TAIL ==========")
    print(df.tail())
    print("\n========== SAMPLE ==========")
    print(df.sample())
    print("\n========== VALUE COUNTS (por columna categórica) ==========")
    col_categoricas =  df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in col_categoricas:
            print(f"\n--- {c} ---")
            print(df[c].value_counts)

    print("\n========== DUPLICADOS ==========")

    print(f"Duplicados: {df.duplicated().sum()}")

    print("\n========== HISTOGRAMAS NUMÉRICOS ==========")

    df.hist(bins=20, figsize=(25,25))
    plt.show()


#%% 
# Cargando limpieza para df_listings
def limpieza (df, name):
    
    print(f"\nlimpieza del dataframe: {name}")
    
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    
    df = df.drop(['name','license','host_name'], axis=1)
    
    return df

#%%
# Cargando función: análisis
def analisis(df_listings, name):
    
    print(f"\n{'='*40}")
    print(f"ANÁLISIS DEL DATAFRAME: {name.upper()}")
    print(f"{'='*40}\n")
    """
    El dataset df_listings contiene 25.000 alojamientos únicos sin registros duplicados, con una oferta
    mayoritariamente concentrada en viviendas completas y en zonas céntricas. Presenta valores nulos
    relevantes en variables clave como el precio y la licencia, así como valores atípicos elevados que
    afectan a la media del precio, por lo que requiere un tratamiento previo antes de realizar análisis
    más profundos, aunque su estructura general es adecuada para continuar con el proceso de EDA.
    """
    print("========== Nº DE ALOJAMIENTOS ÚNICOS ==========")
    # ¿Cuántos alojamientos únicos hay en el dataset?
    
    print(f"\nEl dataset incluye {df_listings['id'].nunique()} alojamientos únicos, de un total de {df_listings.shape[0]} filas totales.\n")
    
    print("========== TIPOS DE ALOJAMIENTO ==========")
    # ¿Qué tipos de alojamientos hay y cuál es el más común?
    
    print(f"\nExisten {df_listings['room_type'].nunique()} tipos de alojamientos únicos:{(df_listings['room_type'].unique())}. \nEl más común es {df_listings['room_type'].value_counts().index[0]} con {df_listings['room_type'].value_counts().iloc[0]} alojamientos.\n")
    
    print("========== DISTRIBUCION ALOJAMIENTOS ==========")
    # ¿Cómo se distribuye los alojamientos por barrio y por zona?
    
    print('\nObservamos que los alojamientos se distribuyen por barrio y por zona:')
    
    alojamientos_por_barrio = df_listings.groupby([df['neighbourhood_group'].str.upper(), 'neighbourhood'])['id'].count().sort_values(ascending=False)
    
    print(alojamientos_por_barrio) 
    
    print('\nLos alojamientos no se distribuyen de forma homogénea por la ciudad: se concentran en distritos específicos, \nmás céntricos o mejor conectados, mientras que zonas periféricas presentan una oferta mucho más limitada. \nEsto sugiere patrones de localización ligados a accesibilidad, atractivo turístico y servicios disponibles.\n')    
    
    alojamientos_por_distrito = df_listings.groupby(['neighbourhood_group'])['id'].count().sort_values(ascending=False)  
    alojamientos_por_distrito.plot(kind='bar', figsize=(15,5))
    plt.title('Número de alojamientos por distrito')
    plt.xlabel('Distrito')
    plt.ylabel('Número de alojamientos')
    plt.xticks(rotation=90)
    plt.show() 
    
    print("========== ZONAS CON MAYOR CONCENTRACION ==========")
    # ¿Existen barrios con una concentración especialmente alta de alojamientos?
    
    df_listings.groupby(['neighbourhood', 'neighbourhood_group'])['id'].count().sort_values(ascending=False).head(10)
    
    print(f"\nExisten barrios con una concentración especialmente alta de alojamientos: \n{df['neighbourhood'].value_counts().head(10)}")
    
    print('\nLos barrios con mayor concentración de alojamientos son aquellos que probablemente sean más turísticos o céntricos, \nlo que puede indicar una mayor demanda y atractivo para los visitantes. \nEsto también puede reflejar la presencia de atracciones turísticas, buena conectividad y servicios disponibles en esos barrios, \nlo que los hace más atractivos para los anfitriones y los huéspedes.\n')
    
    print("========== NULOS ==========")
    #¿Qué variables presentan valores nulos y en qué proporción?
    
    print('\nPorcentaje de nulos por columna:\n')
    
    print(f"{(df_listings.isnull().sum() / df_listings.shape[0] * 100).round(2)}%\n")
    
    print("========== VARIABLES DE BAJO VALLOR ANALÍTICO ==========")
    #¿Hay columnas cuyo contenido no aporta valor al análisis y podrían eliminarse?
    
    print(f"\nComprobando variables con un solo valor único:\n{df_listings.columns[df_listings.nunique() == 1].tolist()} No existen.")
    print(f"\nSin embargo, algunas columnas presentan bajo valor analítico en esta fase exploratoria:")
    print('\nname: texto libre con alta cardinalidad y escaso valor agregado.' ) 
    print('\nlicense: elevado porcentaje de nulos y gran heterogeneidad de formatos.')  
    print('\nhost_name: información nominal sin impacto directo en el análisis exploratorio.')
    print('\nEstas variables podrían eliminarse para este análisis.')
    
    print('\n')
    
    print("========== OUTLIERS ==========")
    #¿Existen valores atípicos en el precio de los alojamientos? ¿Cómo afectan a la media?
    
    print(f"\nExisten valores atípicos en el precio de los alojamientos. \nEl precio medio es {df_listings['price'].mean():.2f}, pero la mediana es {df_listings['price'].median():.2f}, \nlo que indica que los valores atípicos están influyendo en la media y elevándola por encima de la mediana.") 
    print('\nLos valores atípicos en el precio pueden distorsionar la media, haciéndola menos representativa del precio típico de los alojamientos. \nLa mediana, al ser menos sensible a los valores extremos, proporciona una mejor medida de tendencia central en este caso, \nindicando que la mayoría de los alojamientos tienen precios más bajos que la media sugiere debido a la presencia de algunos alojamientos muy caros.')

    print('\n\nEl objetivo de esta fase -entender el contexto general de los datos y detectar posibles problemas de calidad- se considera alcanzado.')


#%%
# Cargando función transformación
def transformacion(df, name):
    
    print(f"Transformacion del dataframe: {name}")
    
    df[['last_rev_year', 'last_rev_month']] = df['last_review'].str.split('-', expand=True).get([0,1])

    df.insert(df.columns.get_loc('last_review')+ 1, 'last_rev_year', df.pop('last_rev_year'))

    df.insert(df.columns.get_loc('last_review')+ 2, 'last_rev_month', df.pop('last_rev_month'))
    
    df['last_rev_year'] = df['last_rev_year'].astype('Int64')

    df['last_rev_month'] = df['last_rev_month'].astype('Int64')
    
    df.drop('last_review', axis=1, inplace=True)
    
    return df

#%%
# Ejecutando transformación
df = transformacion(df, 'listings')

#%%
# Cargando función gestion_nulos
def gestion_nulos(df, name):
    
    print(f"Las columnas con nulos de {name} son de tipo: \n")
    print(df[df.columns[df.isna().any()]].dtypes)
    
    col_num_nul = df.columns[df.isna().any()]
    
    for col in col_num_nul:
        print(f"\nDistribución(%) de las categorías (incluyendo nulos) para la columna", col.upper())
        print(f"{(df[col].value_counts(dropna=False, normalize=True)*100).round(2)} %")
        
    '''Teniendo en cuenta el contexto del análisis, el precio es un eje central, clave para:

        - comparar barrios
        - analizar demanda
        - detectar oportunidades
        - estudiar rentabilidad
        - segmentar alojamientos

    Un 24% de nulos en esta variable es un porcentaje altísimo. 
    Imputarlos supondría inventar el precio de unos 6000 alojamientos, 1 de cada 4, 
    pero eliminarlos supone perder el 24% del dataset. 
    
    Vamos a comprobar la distribución del precio antes de tomar una decisión.
    ''' 

#%% 
# Cargando función analisis_previo_imputacion
def analisis_previo_imputacion(df):
    """
    Realiza un análisis estadístico de los precios y elimina registros de 
    alojamientos inactivos (sin precio y disponibilidad 0).

    Args:
        df (pd.DataFrame): DataFrame a procesar.

    Returns:
        pd.DataFrame: DataFrame con los alojamientos inactivos eliminados.
    """

    print('======= ESTADISTICOS DE PRICE =========')
    
    stats_price = df['price'].describe().round(2)
    print(f"\n{stats_price}")
    
    print('\nVemos precios muy dispersos, con un valor mínimo de 8€ y un valor máximo de 25654€.\n Lo veremos mejor en un boxplot')
    
    plt.boxplot(df['price'].dropna())
    plt.title(f"Boxplot de Precios")
    plt.ylabel('price')
    plt.show()
    
    print('\nEl boxplot muestra una distribución de precios muy asimétrica, con numerosos valores atípicos en el extremo superior. ')
    print('\nEsto indica la presencia de alojamientos con precios extremadamente altos que distorsionan la media.\n Para análisis posteriores sería recomendable aplicar técnicas de tratamiento de outliers como recorte por percentiles o transformación logarítmica.')
    
    print('Vamos a ver cuantos registros nulos en el precio tienen valor 0 en availability')

    # 1) Máscara: price nulo Y disponibilidad 0
    mask_inactivos = df['price'].isna() & (df['availability_365'] == 0)

    # 2) Número de filas inactivas
    apt_inactivos = mask_inactivos.sum()

    # 3) Tasa sobre el total del dataset
    tasa_total = apt_inactivos / df.shape[0] * 100


    print(f"{apt_inactivos} registros tienen PRICE nulo y availability_365 = 0 (apartamentos inactivos).")
    print(f"La tasa de apartamentos inactivos sobre el total de registros del dataset es de {tasa_total.round(2)}%.\nLo más razonable es prescindir de estos registros para evitar sesgos irreales en nuestro análisis.")

    print('Eliminando registros de apartamentos inactivos...')
    
    df = df[~mask_inactivos].copy()
    
    print(f"{apt_inactivos} registros eliminados")
    
    print('Comprobamos los nulos de nuevo')
    
    col_num_nul = df.columns[df.isna().any()]
    
    for col in col_num_nul:
        print(f"\nDistribución(%) de las categorías (incluyendo nulos) para la columna", col.upper())
        print(f"{(df[col].value_counts(dropna=False, normalize=True)*100).round(2)} %")
    
    return df
    
# %% 
# cargando función: imputacion_nulos
def imputacion_nulos(df):
    
    print(f"\nDebido al alto número de outliers imputamos nulos en PRICE usando la mediana")
    print('\nImputando nulos en PRICE...')
          
    df['price'] = df['price'].fillna(df['price'].median())
    
    print(f"\nNulos en PRICE: {df['price'].isna().sum()}")
    
    print('===========')
    
    reviews_col = df[['last_rev_year', 'last_rev_month','reviews_per_month']].columns.tolist()
    
    for col in reviews_col:
        
        print(f"\nPara la imputación de nulos en {col} crearemos un nuevo valor = 0")
        
        print(f"\nImputando nulos en {col}...")
        
        df[col] = df[col].fillna(0)
        
        print(f"\nNulos en {col}: {df['price'].isna().sum()}")
        
        
    return df


# %%
# cargando funcion analisis_precio
def analisis_precio(df):
    
    print('=========== ANÁLISIS DE PRECIO ===============')

    print('\n¿Cuál es el precio medio y mediano de los alojamientos?\n')
    print('='*40)
    print(df['price'].agg(['mean', 'median', 'std', 'max', 'min']))
    print('\nDebido a la presencia de valores extremos que distorsionan la visualización, para algunos análisis gráficos se excluirán outliers \nmediante el rango intercuartílico (IQR), lo que permite una comparación más representativa de los precios típicos entre categorías.')

    print('________________________________')

    print('\n¿Cómo varía el precio según el tipo de alojamiento?')
    print('='*40)
    # Rango Intercuartílico (IQR) global para filtrar outliers
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    limite_inf, limite_sup

    # Dataset sin outliers (solo para análisis/visualización)
    df_iqr = df[(df['price'] >= limite_inf) & (df['price'] <= limite_sup)].copy()

    # Resumen por tipo de alojamiento: mediana, Q1, Q3 e IQR
    resumen_iqr = (df_iqr.groupby('room_type')['price'].agg(mediana='median',Q1=lambda x: x.quantile(0.25),Q3=lambda x: x.quantile(0.75)))
    resumen_iqr['IQR'] = resumen_iqr['Q3'] - resumen_iqr['Q1']

    print(resumen_iqr)

    # Gráfico del precio típico por tipo de alojamiento (sin outliers)
    plt.figure()
    resumen_iqr['mediana'].plot(kind='bar')
    plt.ylabel('Precio mediano (€)')
    plt.title('Precio típico por tipo de alojamiento (sin outliers)')
    plt.xticks(rotation=0)
    plt.show()

    print('\nEl precio del alojamiento varía significativamente según el tipo de habitación, \nsiendo los alojamientos completos los más caros y los compartidos los más económicos. \nLa presencia de valores atípicos, especialmente en apartamentos completos y habitaciones privadas, \nprovoca que la media sea superior a la mediana, por lo que esta última resulta una medida más representativa del precio típico.')

    # Gráfico del IQR por tipo de alojamiento
    plt.figure()
    resumen_iqr['IQR'].sort_values(ascending=False).plot(kind='bar')
    plt.ylabel('IQR (€)')
    plt.title('Dispersión del precio (IQR) por tipo de alojamiento (sin outliers)')
    plt.xticks(rotation=0)
    plt.show()

    print('\nEl análisis del rango intercuartílico (IQR) del precio por tipo de alojamiento muestra que los alojamientos completos y \nlas habitaciones de hotel presentan la mayor dispersión de precios, lo que indica una elevada heterogeneidad dentro de estos segmentos. \nPor el contrario, las habitaciones compartidas exhiben una variabilidad muy reducida, reflejando precios más homogéneos y predecibles.')

    # Gráfico de la relación entre precio mediano y dispersión
    plt.figure()
    plt.scatter(resumen_iqr['mediana'], resumen_iqr['IQR'])

    for room_type, row in resumen_iqr.iterrows():
        plt.text(row['mediana'], row['IQR'], room_type, ha='left', va='bottom')

    plt.xlabel('Precio mediano (€)')
    plt.ylabel('Dispersión (IQR €)')
    plt.title('Relación entre precio mediano y dispersión (sin outliers)')
    plt.show()

    print('\nEl análisis conjunto del precio mediano y el rango intercuartílico muestra una relación positiva entre nivel de precio y dispersión. \nLos tipos de alojamiento más caros presentan mayor variabilidad, mientras que los más económicos muestran precios más estables y predecibles.')
    print('\n________________________________')


    print('¿Existen diferencias significativas de precio entre barrios?')
    print('='* 40)

    # Mediana del precio por distrito y barrio
    precio_barrio = (df_iqr.groupby(['neighbourhood_group', 'neighbourhood'])['price'].median().reset_index())

    distritos = precio_barrio['neighbourhood_group'].unique()
    n = len(distritos)

    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))
    axes = axes.flatten()

    for i, distrito in enumerate(distritos):
        df_distrito = (precio_barrio[precio_barrio['neighbourhood_group'] == distrito].sort_values('price', ascending=False))
        axes[i].bar(df_distrito['neighbourhood'], df_distrito['price'])
        axes[i].set_title(distrito)
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_ylabel('€')

    # Eliminar ejes vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Precio mediano por barrio y distrito (sin outliers)', fontsize=16)
    plt.tight_layout()
    plt.show()

    print('\nLa comparación del precio mediano por barrio dentro de cada distrito muestra una notable heterogeneidad interna: \nincluso en distritos con precios elevados, coexisten barrios con niveles de precio sensiblemente distintos, \nlo que confirma que el barrio introduce diferencias relevantes en el precio del alojamiento más allá del distrito al que pertenece.')
    print('\n________________________________')


    print('¿Qué barrios presentan los precios más altos y cuáles los más bajos?')
    print('='* 40)
    #Calculamos la mediana del precio por barrio
    precio_barrio = df_iqr.groupby(['neighbourhood', 'neighbourhood_group'], as_index=False)['price'].median().rename(columns={'price':'price_median'})

    # Seleccionamos extremos
    barrios_mas_baratos = precio_barrio.nsmallest(5,'price_median')
    barrios_mas_caros = precio_barrio.nlargest(5, 'price_median')

    # Unimos
    extremos = pd.concat([barrios_mas_baratos, barrios_mas_caros])

    plt.figure()

    sns.barplot(data=extremos,x='neighbourhood',y='price_median',hue='neighbourhood_group')

    plt.xticks(rotation=90)
    plt.xlabel('Barrio')
    plt.ylabel('Precio mediano (€)')
    plt.title('Barrios con precios más bajos y más altos (precio mediano)')
    plt.show()
    print('La comparación del precio mediano por barrio evidencia una clara brecha territorial en el mercado del alojamiento: \nlos barrios periféricos, como Vinateros, Arcos, Campamento, Fontarrón y Los Ángeles, presentan los precios medianos más bajos, \nmientras que los barrios centrales y de mayor renta —especialmente Castellana, Recoletos, Lista, Niño Jesús y Sol— \nconcentran los precios medianos más elevados, confirmando la fuerte influencia de la localización en el nivel de precios del alojamiento.\n')

    print(df.groupby(['neighbourhood_group', 'neighbourhood'])['price'].agg(['min', 'max']).sort_values('max', ascending=False))

    print('Los precios más altos se concentran en barrios céntricos como Sol y Universidad, donde aparecen valores extremos muy elevados. \nEn contraste, los barrios periféricos como Horcajo, Cuatro Vientos o Santa Eugenia presentan precios notablemente más bajos \ny rangos más homogéneos, lo que indica mercados más estables y sin presencia de outliers significativos.')
    print('\n________________________________')

    print('\n¿Qué tipo de alojamiento ofrece la mejor relación entre precio y valoración?')
    print('='* 40)
    res = df.groupby('room_type')[['price','number_of_reviews_ltm']].median()
    res['reviews_por_euro'] = res['number_of_reviews_ltm'] / res['price']
    print(res.sort_values('reviews_por_euro', ascending=False))
    print('Los alojamientos compartidos ofrecen la mejor relación entre precio y valoración, al presentar el mayor número de reseñas \npor euro pagado. A medida que aumenta el nivel de privacidad y el precio del alojamiento, esta relación disminuye, \nsiendo las habitaciones de hotel las menos eficientes en términos de valoración por coste.')

    # Correlación

    print('CORRELACIÓN DE KENDALL')

    corr_kendall_room = (
        df
        .groupby('room_type')[['price', 'number_of_reviews']]
        .corr(method='kendall')
        .iloc[0::2, -1]
    )

    print(corr_kendall_room)

    print('\nEl análisis de correlación de Kendall entre el precio y el número de reseñas, realizado por tipo de alojamiento y excluyendo valores atípicos, \nmuestra una relación débil y predominantemente negativa en todos los tipos de alojamiento. \nEsto indica que, aunque el precio influye en la actividad de valoración, su efecto es limitado y no estrictamente lineal.') 

    plt.figure(figsize=(6,4))
    sns.heatmap(
        corr_kendall_room.to_frame(),
        annot=True,
        cmap='coolwarm',
        center=0,
        cbar_kws={'label': 'Kendall τ'}
    )
    plt.title('Correlación Kendall entre precio y reseñas\npor tipo de alojamiento')
    plt.xlabel('Variable')
    plt.ylabel('Tipo de alojamiento')
    plt.tight_layout()
    plt.show() 
    print('\n________________________________')


    print('¿Existen alojamientos con precios extremadamente altos o bajos? ¿A qué podrían deberse?') 
    print('='* 40)

    # Media y desviación estándar
    media = df['price'].mean()
    std = df['price'].std()

    # Límites de la regla empírica
    limite_inf_emp = media - 3 * std
    limite_sup_emp = media + 3 * std

    # Máscara de outliers según la regla empírica
    mask_empirica = (df['price'] < limite_inf_emp) | (df['price'] > limite_sup_emp)

    # DataFrames resultantes
    df_empirica = df[~mask_empirica]     # sin outliers
    df_outliers_emp = df[mask_empirica]  # solo outliers

    print("Outliers detectados:", df_outliers_emp.shape[0])

    df_outliers_emp['price'].agg(['max', 'min'])

    print('La aplicación de la regla empírica (media ± 3 desviaciones estándar) identifica un número reducido de valores atípicos, \nconcentrados exclusivamente en el extremo superior de la distribución del precio. Aunque el límite inferior resulta negativo \n—y por tanto no interpretable en términos de precio—, los outliers detectados corresponden a alojamientos con precios excepcionalmente elevados, \nque alcanzan valores máximos muy superiores al rango habitual del mercado. \nEstos precios extremos pueden deberse a alojamientos de características singulares, estrategias de fijación de precios atípicas \no posibles errores en el registro de los datos.') 
    

#%%
# Cargando función: análisis_disponibilidad
def analisis_disponibilidad(df):
    """
    Analiza la disponibilidad y el comportamiento de la oferta de alojamientos,
    incluyendo su distribución anual, variaciones por barrio y tipo de alojamiento,
    y su relación con el precio y la demanda.

    Args:
        df (pd.DataFrame): DataFrame con la información de los alojamientos.
    """

    print("\nDISPONIBILIDAD Y COMPORTAMIENTO DE LA OFERTA")
    print("=" * 55)

    # --- 1) Distribución de la disponibilidad anual ---
    print("\n¿Cómo se distribuye la disponibilidad anual de los alojamientos?")
    print("-" * 55)
    print(df['availability_365'].describe())

    plt.figure()
    df['availability_365'].plot(kind='hist', bins=30)
    plt.xlabel('Disponibilidad anual (días)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de availability_365')
    plt.show()

    # --- 2) Barrios con mayor disponibilidad media ---
    print("\n¿Existen barrios con mayor disponibilidad media?")
    print("-" * 55)
    disp_barrio = (
        df.groupby(['neighbourhood_group', 'neighbourhood'])['availability_365'].mean().sort_values(ascending=False))
    print(disp_barrio.head(15))

    # Barplot Top 15 barrios por disponibilidad media
    top_disp = disp_barrio.head(15).reset_index()
    top_disp.columns = ['neighbourhood_group', 'neighbourhood', 'availability_mean']

    plt.figure()
    plt.bar(top_disp['neighbourhood'], top_disp['availability_mean'])
    plt.xticks(rotation=90)
    plt.xlabel('Barrio')
    plt.ylabel('Disponibilidad media (días)')
    plt.title('Top 15 barrios con mayor disponibilidad media')
    plt.show()

    # --- 3) Tipo de alojamiento con mayor disponibilidad ---
    print("\n¿Qué tipo de alojamiento tiende a tener mayor disponibilidad?")
    print("-" * 55)
    disp_room = df.groupby('room_type')['availability_365'].mean().sort_values(ascending=False)
    print(disp_room)

    plt.figure()
    disp_room.plot(kind='bar')
    plt.xticks(rotation=0)
    plt.xlabel('Tipo de alojamiento')
    plt.ylabel('Disponibilidad media (días)')
    plt.title('Disponibilidad media por tipo de alojamiento')
    plt.show()

    # --- 4) Relación con precio y demanda (reseñas LTM si existe) ---
    print("\n¿Puede la disponibilidad estar relacionada con el precio o la demanda?")
    print("-" * 55)

    cols = ['availability_365', 'price']
    if 'number_of_reviews_ltm' in df.columns:
        cols.append('number_of_reviews_ltm')
    elif 'number_of_reviews' in df.columns:
        cols.append('number_of_reviews')

    # Correlación Spearman (robusta para no-linealidad)
    corr = df[cols].corr(method='spearman')
    print("\nCorrelación (Spearman):")
    print(corr)

    # Scatter disponibilidad vs precio
    plt.figure()
    plt.scatter(df['availability_365'], df['price'], alpha=0.35)
    plt.xlabel('Disponibilidad anual (días)')
    plt.ylabel('Precio (€)')
    plt.title('Disponibilidad vs Precio')
    plt.show()

    # Scatter disponibilidad vs demanda (si hay columna)
    if 'number_of_reviews_ltm' in df.columns:
        demanda_col = 'number_of_reviews_ltm'
        plt.figure()
        plt.scatter(df['availability_365'], df[demanda_col], alpha=0.35)
        plt.xlabel('Disponibilidad anual (días)')
        plt.ylabel('Reseñas (LTM)')
        plt.title('Disponibilidad vs Demanda (Reseñas LTM)')
        plt.show()
    elif 'number_of_reviews' in df.columns:
        demanda_col = 'number_of_reviews'
        plt.figure()
        plt.scatter(df['availability_365'], df[demanda_col], alpha=0.35)
        plt.xlabel('Disponibilidad anual (días)')
        plt.ylabel('Número de reseñas')
        plt.title('Disponibilidad vs Demanda (Reseñas)')
        plt.show()

    print("\n________________________________")
    

