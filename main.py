# main.py 
import pandas as pd 

# importamos funciones 
from funciones import (
    eda,
    limpieza,
    analisis, 
    transformacion, 
    gestion_nulos,
    analisis_previo_imputacion,
    imputacion_nulos,
    analisis_precio,
    analisis_disponibilidad)

def main():
    """
    Función principal que orquesta el flujo de carga, limpieza, análisis y
    transformación de los datos de AIRBNB.
    """
    
    # cargando datos
    df = pd.read_csv('listings.csv')
    
    # llamar a funciones y capturar resultados para persistir cambios
    eda(df, 'LISTINGS')
    df = limpieza(df, 'LISTINGS')
    analisis(df, 'LISTINGS')
    df = transformacion(df, 'LISTINGS')
    
    # Estas funciones imprimen resultados o modifican/devuelven el DF
    gestion_nulos(df, 'LISTINGS')
    df = analisis_previo_imputacion(df)
    df = imputacion_nulos(df)
    analisis_precio(df)
    analisis_disponibilidad(df)

if __name__ == "__main__":
    main()
