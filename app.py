import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Walmart Sales AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para "pulir" la estética
st.markdown("""
    <style>
    .big-font { font-size:50px !important; color: #4CAF50; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Intenta cargar el archivo localmente
        return joblib.load('modelo_ventas_walmart_final.joblib')
    except Exception as e:
        return None

model = load_model()

# --- 3. SIDEBAR (CONTROLES) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0b/Walmart_logo_%282025%3B_Alt%29.svg", width=150)
    st.header("⚙️ Configuración")
    
    # Datos Temporales
    st.subheader("1. Fecha y Tienda")
    store = st.selectbox("Seleccionar Tienda", list(range(1, 46)))
    date_val = st.date_input("Fecha a Proyectar", datetime.date(2012, 11, 2))
    
    holiday_opt = st.radio("¿Es Semana Festiva?", ["No", "Sí"], horizontal=True)
    is_holiday = 1 if holiday_opt == "Sí" else 0

    st.markdown("---")
    
    # Datos Económicos (Usamos expander para limpiar la vista)
    with st.expander("2. Indicadores Económicos", expanded=True):
        temp = st.slider("Temperatura (°F)", -10, 105, 60, help="Temperatura promedio de la región")
        fuel = st.number_input("Precio Combustible ($)", 2.0, 5.0, 3.60, step=0.1)
        cpi = st.number_input("CPI (Índice Precios)", 100.0, 230.0, 190.0, help="Consumer Price Index")
        unemp = st.slider("Tasa Desempleo (%)", 3.0, 15.0, 8.0, step=0.1)

# --- 4. LÓGICA DE PREDICCIÓN ---
def make_prediction():
    # Feature Engineering en tiempo real (Igual que en el entrenamiento)
    year = date_val.year
    month = date_val.month
    week = date_val.isocalendar()[1]
    
    # DataFrame con nombres de columnas EXACTOS
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
    
    if model:
        prediction = model.predict(input_data)[0]
        return prediction, input_data
    return 0, input_data

# --- 5. INTERFAZ PRINCIPAL ---
st.title("📈 Proyección de Ventas con IA")
st.markdown("### Maestría en Inteligencia Artificial Aplicada")

if model is None:
    st.error("⚠️ Error Crítico: No se encuentra el archivo .joblib. Verifica que subiste el modelo al repositorio.")
else:
    # Generar predicción automática al cambiar inputs
    pred_val, input_df = make_prediction()
    
    # Pestañas para organizar la información
    tab1, tab2 = st.tabs(["📊 Dashboard Principal", "📝 Detalles del Modelo"])

    with tab1:
        # Columna Izquierda: Resultado Numérico
        col_res, col_graph = st.columns([1, 2])
        
        with col_res:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("Ventas Estimadas:")
            st.markdown(f'<p class="big-font">${pred_val:,.2f}</p>', unsafe_allow_html=True)
            
            # Semáforo de Ventas
            if pred_val > 2000000:
                st.success("🔥 Desempeño: EXCEPCIONAL")
            elif pred_val > 1000000:
                st.info("✅ Desempeño: PROMEDIO")
            else:
                st.warning("📉 Desempeño: BAJO")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.write("")
            st.caption(f"Semana: {input_df['Week'][0]} | Año: {input_df['Year'][0]}")

        with col_graph:
            # Gráfico de Contexto (Termómetro)
            st.write("#### Contexto de Mercado")
            
            # Datos simulados de referencia (Min, Promedio, Max históricos aproximados)
            referencias = {'Mínimo Histórico': 200000, 'Tu Predicción': pred_val, 'Promedio': 1050000, 'Máximo Histórico': 3000000}
            df_ref = pd.DataFrame.from_dict(referencias, orient='index', columns=['Ventas'])
            
            # Crear gráfico
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['gray', '#4CAF50', 'blue', 'gray']
            sns.barplot(x=df_ref['Ventas'], y=df_ref.index, palette=colors, ax=ax)
            ax.set_xlabel("Ventas ($)")
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    with tab2:
        st.write("### Datos de Entrada al Modelo")
        st.dataframe(input_df)
        st.write("### Sobre el Modelo")
        st.json({
            "Algoritmo": "Random Forest Regressor",
            "Librería": "Scikit-Learn",
            "Entrenamiento": "Datos históricos 2010-2012",
            "Variables Clave": ["Tienda", "Semana", "CPI", "Desempleo"]
        })






