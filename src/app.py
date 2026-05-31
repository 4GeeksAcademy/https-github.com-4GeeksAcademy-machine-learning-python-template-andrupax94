import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "models" / "models_pack.pkl"
JUPITER_PATH = BASE_DIR / ".." / "notebooks" / "09-k-means-housing.html"
CSV_PATH = BASE_DIR / ".." / "data" /"raw"/ "housing.csv"
VIRIDIS_6 = {
    0: "#440154D2",  # morado oscuro
    1: "#404387D2",  # azul-morado
    2: "#29788ED2",  # azul-verde
    3: "#22A784D2",  # verde-azul
    4: "#79D151D2",  # verde-amarillo
    5: "#FDE724D2",  # amarillo
}
# configuracion de la pagina
st.set_page_config(page_title="App de Viviendas", layout="wide")

# --- COMPONENTE PARA BACKGROUND ---
# aqui puedes meter tu link de canva o css personalizado
# lo dejo listo para que solo cambies el color o la imagen
st.markdown(
    """
    <style>
    .stApp {
        background: #000000; /* cambia esto por url('tu_imagen_de_canva') si kieres */
        background-size: cover;
    }
    /* un pokito de estilo para que no parezca tan de fabrica */
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- NAVBAR SIMPLE ---
# usamos un radio button en el sidebar para que parezca un menu
st.sidebar.title("Menu de Navegacion")
opcion = st.sidebar.radio("Ir a:", ["Mi Aplicacion", "Ver Notebook (Jupiter)"])

if opcion == "Mi Aplicacion":
    # --- TU APP ORIGINAL CON RUTAS NUEVAS ---
    
    # cargamos los modelos (rutas relativas desde src/)
    try:
        model_pack = joblib.load(MODEL_PATH)
        kmeans = model_pack["model_km"]
        classifier = model_pack["model_rf"]
    except:
        st.error("No se encuentran los modelos en ../models/ checka las carpetas")

    st.title("🏠 Mi App de Casas en California")
    st.write("Hola, esta es una aplicacion para ver en que grupo cae tu casa segun la zona y cuanto ganas.")
    st.write("Lo hice para clase, espero que funcione bien...")

    st.header("Pon los datos de tu casa aki")

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitud (ej: 34.0)", value=34.0)
        lon = st.number_input("Longitud (ej: -118.0)", value=-118.0)
        ingreso = st.number_input("Ingreso Medio (en miles)", value=3.0)

    if st.button("Predecir Grupo"):
        datos = np.array([[lat, lon, ingreso]])
        pred = classifier.predict(datos)[0]
        st.success(f"Tu casa pertenece al grupo: {pred}")
        st.write("He usado el modelo que entrenamos en el jupiter.")

    st.header("Mapa de las casas")
    st.write("Aqui puedes ver donde estan las casas del ejemplo y la tuya")

    # cargar datos de muestra (ruta nueva)
    try:
        
        df = pd.read_csv(CSV_PATH)

        # Mismo proceso que el notebook:
        # 1) Escalar con StandardScaler
        # 2) Predecir clusters con el kmeans guardado
        # Las features que usa kmeans son todas las columnas menos MedHouseVal
        # (target), igual que en el notebook donde X = df.drop('MedHouseVal')
        features = [ 'Latitude', 'Longitude','MedInc']

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        df["cluster"] = kmeans.predict(X_scaled).astype(int)

        # Color viridis por cluster — columna con listas [R,G,B,A]
        df["color"] = df["cluster"].apply(lambda c: VIRIDIS_6[c])
        df = df[['Latitude', 'Longitude','color']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon','color':'col'})
        st.map(df,color="col")
    except:
        st.warning("No se pudo cargar el mapa, falta el csv en ../data/raw/")

    st.write("---")
    st.write("Recursos usados: streamlit, pandas, sklearn y mucha paciencia.")
    st.write("No me juzguen por el diseño, no soy diseñador.")

elif opcion == "Ver Notebook (Jupiter)":
    st.title("📓 Visualizacion del Notebook")
    st.write("Este es el trabajo previo de analisis que hice en Jupiter.")
    
    # leer el html generado
  
    if os.path.exists(JUPITER_PATH):
        with open(JUPITER_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # insertar el html en un componente
        components.html(html_content, height=800, scrolling=True)
    else:
        st.error("No se encontro el archivo HTML del notebook. Tienes que generarlo primero.")
