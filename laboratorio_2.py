import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Laboratorio 2")

capital_inicial = 20000
años = [i for i in range(1,18)]


def rendimiento_anual(rendimiento):
    return capital_inicial * (1 + rendimiento)

def cumsum(array):
    cumsum_values=[]
    for index, val in enumerate(array):
        cumsum_values[index] = cumsum_values[index] + val
    return cumsum_values


rendimiento_promedio = st.number_input("Tasa de rendimiento promedio (%)")
desviacion = st.number_input("Desviacion estandar (%)") 
rendimientos = []
montos_finales = []
dataset_simulacion_rendimiento = pd.DataFrame(columns=años)
dataset_simulacion_monto = pd.DataFrame(columns=años)

numero_simulaciones = st.number_input("Cantidad de simulaciones")

for i in range(0, int(numero_simulaciones)):
    rendimientos_actuales = np.random.normal(rendimiento_promedio / 100, desviacion/100, 17)
    capital_final = [rendimiento_anual(x) for x in rendimientos_actuales]
    rendimiento_promedio_actual = np.mean(rendimientos_actuales)
    rendimientos.append(rendimiento_promedio_actual)
    monto_final = np.sum(capital_final)
    montos_finales.append(monto_final)
    dataset_simulacion_rendimiento.loc[len(dataset_simulacion_rendimiento)] = rendimientos_actuales
    dataset_simulacion_monto.loc[len(dataset_simulacion_monto)] = capital_final

if len(rendimientos) > 0:
    rendimientos_promedio_por_año = (list(dataset_simulacion_rendimiento.mean(axis=0)))
    monto_promedio_por_año = (list(dataset_simulacion_monto.mean(axis=0)))
    suma_por_iteracion = list(dataset_simulacion_monto.sum(axis=1))
    mejores_rendimientos = dataset_simulacion_rendimiento.iloc[np.argmax(suma_por_iteracion), :]
    peores_rendimientos = dataset_simulacion_rendimiento.iloc[np.argmin(suma_por_iteracion), :]
    montos_acumulados = np.cumsum(monto_promedio_por_año)
    rendimiento_promedio_simulacion = np.mean(rendimientos)
    monto_promedio_simulacion = np.mean(montos_finales)

    '''### Rendimiento promedio total al finalizar los 17 años.'''
    st.write(f'''{round(rendimiento_promedio_simulacion, 2) * 100}%''')
    '''### Monto promedio acumulado al finalizar los 17 años.'''
    st.write(f'''{round(monto_promedio_simulacion, 2)}''')
    '''### Escenario de ahorro pesimista y optimista.'''
    st.write(f'''El mejor escenario es {np.argmax(suma_por_iteracion)} con un monto de  {round(np.max(suma_por_iteracion),2)}''')
    st.table(np.array(mejores_rendimientos))
    st.write(f'''El peor escenario es {np.argmin(suma_por_iteracion)} con un monto de  {round(np.min(suma_por_iteracion),2)}''')
    st.table(np.array(peores_rendimientos))
    
    '''### Gráfica de los rendimientos obtenidos por cada año.'''
    plot = sns.barplot(x=años, y=rendimientos_promedio_por_año)
    plt.title('Rendimiento Promedio por Año')
    plt.xlabel("Año")
    plt.ylabel("Rendimiento")
    st.pyplot(plot.get_figure())
    '''### Grafica del monto ahorrado por cada año.'''
    plot = sns.barplot(x=años, y=monto_promedio_por_año)
    plt.title('Monto Promedio por Año')
    plt.xlabel("Año")
    plt.ylabel("Ahorro Promedio")
    st.pyplot(plot.get_figure())
    '''###  Grafica del monto acumulado por cada año dentro de los 17 años.'''
    plot = sns.barplot(x=años, y=montos_acumulados)
    plt.title('Montos Acumulados')
    plt.xlabel("Año")
    plt.ylabel("Ahorro Anual")
    st.pyplot(plot.get_figure())
     
    






