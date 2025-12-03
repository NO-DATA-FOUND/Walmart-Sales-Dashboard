import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción Walmart", page_icon="🛒")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    # Buscamos el modelo en la carpeta 'modelo'
    ruta_modelo = os.path.join("walmart_ventas_model_final.joblib")
    try:
        return joblib.load(ruta_modelo)
    except FileNotFoundError:
        st.error("⚠️ Error: No se encuentra el archivo del modelo en la carpeta 'modelo/'.")
        return None

model = load_model()

# --- INTERFAZ ---
st.title("📈 Dashboard de Predicción de Ventas")
st.markdown("### Maestría en Inteligencia Artificial Aplicada")
st.info("Este prototipo estima las ventas semanales utilizando un modelo Random Forest optimizado.")

# --- SIDEBAR (ENTRADAS) ---
st.sidebar.header("Parámetros de Entrada")

# 1. Datos de Tienda y Fecha
store = st.sidebar.selectbox("Número de Tienda", list(range(1, 46)))
date_val = st.sidebar.date_input("Fecha de Proyección", datetime.date(2012, 11, 2))
holiday = st.sidebar.radio("¿Es Semana Festiva?", ["No", "Sí"])

st.sidebar.markdown("---")

# 2. Datos Económicos
temp = st.sidebar.slider("Temperatura (°F)", -10, 100, 60)
fuel = st.sidebar.number_input("Precio Combustible ($)", 2.0, 6.0, 3.5)
cpi = st.sidebar.number_input("CPI (Índice Precios)", 100.0, 250.0, 190.0)
unemp = st.sidebar.number_input("Tasa Desempleo (%)", 1.0, 15.0, 8.0)

# --- BOTÓN Y CÁLCULO ---
if st.button("Calcular Predicción", type="primary"):
    if model:
        # Preprocesamiento en vivo (Feature Engineering)
        year = date_val.year
        month = date_val.month
        week = date_val.isocalendar()[1]
        is_holiday = 1 if holiday == "Sí" else 0
        
        # DataFrame con las columnas exactas que espera el modelo
        input_data = pd.DataFrame({
            'Temperature': [temp],
            'Fuel_Price': [fuel],
            'CPI': [cpi],
            'Unemployment': [unemp],
            'Year': [year],
            'Month': [month],
            'Week': [week],
            'Store': [store],
            'Holiday_Flag': [is_holiday]
        })
        
        # Predicción
        pred = model.predict(input_data)[0]
        
        # Mostrar Resultado
        st.success("✅ Predicción Generada Exitosamente")
        st.metric(label="Ventas Semanales Estimadas", value=f"${pred:,.2f}")
        
        # Interpretación básica
        if pred > 1500000:
            st.warning("Nota: Se proyecta un volumen de ventas ALTO.")
        else:

            st.info("Nota: Volumen de ventas dentro del rango estándar.")

