#Imporamos Librewrias
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
import pydeck as pdk
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

#Declaramos de una vez los path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "models" / "models_pack.pkl"
JUPITER_PATH = BASE_DIR / ".." / "notebooks" / "09-k-means-housing.html"
CSV_PATH = BASE_DIR / ".." / "data" /"raw"/ "housing.csv"

#declaramos los colores viridis para la leyenda del mapa
VIRIDIS_6 = {
    0: "#440154D2",  
    1: "#404387D2",  
    2: "#29788ED2",  
    3: "#22A784D2",  
    4: "#79D151D2",  
    5: "#FDE724D2",  
}

# configuracion basica de la pagina
st.set_page_config(page_title="Housing 4Geeks", layout="wide")

#Css para algo de estilo al backround
st.markdown(
    """
    <style>
    .stApp {
        background: #111111; /* cambia esto por url('tu_imagen_de_canva') si kieres */
        background-size: cover;
    }
    .main {
        background-color: #111111;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# hacemos un pequeño NavBar
st.sidebar.title("Menu de Navegacion")
opcion = st.sidebar.radio("Ir a:", ["Ver Notebook","Predictor"])


if opcion == "Predictor":
    #intentamos importar el modelo que como recordaremos en el jupiter guardamos 2 en un pack
    try:
        model_pack = joblib.load(MODEL_PATH)
        kmeans = model_pack["model_km"]
        classifier = model_pack["model_rf"]
    except:
        st.error("No se encuentran los modelos")

    st.title("Housing Proyect 4geeks")
    st.write("Predictor en streamlib para el proytecto de prediccion de casas de california")

    st.header("Pon los datos de tu casa aqui")

    col1, col2 = st.columns(2)
    # declaramos las variable del formulario
    with col1:
        lat = st.number_input("Latitud (ej: 34.0)", value=34.0)
        lon = st.number_input("Longitud (ej: -118.0)", value=-118.0)
        ingreso = st.number_input("Ingreso Medio (en miles)", value=3.0)

    # Inicializamos session_state si no existe
    if 'pred_lat' not in st.session_state:
        st.session_state['pred_lat'] = None
        st.session_state['pred_lon'] = None
        st.session_state['pred_cluster'] = None
    #condicional para predecir
    if st.button("Predecir Grupo"):
        datos = np.array([[lat, lon, ingreso]])
        pred = classifier.predict(datos)[0]
        st.success(f"Tu casa pertenece al grupo: {pred}")
        st.session_state["pred_lat"] = lat
        st.session_state["pred_lon"] = lon
        st.session_state["pred_cluster"] = pred

    st.header("Mapa de las casas")
    st.write("Aqui puedes ver donde estan las casas del dataset y la predicha")

    # cargarmos los datos de muestra (ruta nueva)
    try:
        
        df = pd.read_csv(CSV_PATH)
        # Basicamente el mismo proceso que el notebook:
        # 1) Escalar con StandardScaler
        # 2) Predecir clusters con el kmeans guardado
        # Las features que usa kmeans son todas las columnas menos MedHouseVal
        # (target), igual que en el notebook donde X = df.drop('MedHouseVal')
        features = [ 'Latitude', 'Longitude','MedInc']

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        df["cluster"] = kmeans.predict(X_scaled).astype(int)
        #Le asignamos un color mapeando con apply
        df["color"] = df["cluster"].apply(lambda c: VIRIDIS_6[c])
        df = df[['Latitude', 'Longitude','color']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon','color':'color_code'})

        # Convertimos hexadecimal a RGB para pydeck
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return [int(h[i:i+2], 16) for i in (0, 2, 4)] + [210] # Añadir alpha
        
        df['color_rgb'] = df['color_code'].apply(hex_to_rgb)

        # Capa de puntos (ScatterplotLayer)
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_color="color_rgb",
                get_radius=1000,
                pickable=True,
            )
        ]

        # Aca Configuracion de la vista
        if st.session_state['pred_lat'] is not None:
            # Si hay predicción, centrar con mas zoom
            view_state = pdk.ViewState(
                latitude=st.session_state['pred_lat'],
                longitude=st.session_state['pred_lon'],
                zoom=12,
                pitch=0,
            )
            # Añadimos marcador de prediccion
            user_pred_df = pd.DataFrame({
                "lat": [st.session_state['pred_lat']],
                "lon": [st.session_state['pred_lon']]
            })
            
            # Capa para el marcador de la predicción (Punto rojo destacado)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=user_pred_df,
                    get_position=["lon", "lat"],
                    get_color=[255, 0, 0, 255], # Rojo puro
                    get_radius=1000,
                    pickable=True,
                )
            )
        else:
            # Vista general si no hay predicción
            view_state = pdk.ViewState(
                latitude=df["lat"].mean(),
                longitude=df["lon"].mean(),
                zoom=5,
                pitch=0,
            )

        # Renderizamos mapa con pydeck
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=None,
            tooltip={"text": "Coordenadas: {lat}, {lon}"}
        ))
        #estilos para la leyenda
        html_legend_style="""
        <style>
            .legend_container{
                display:flex;
                
                }
            .color_container{
                display:flex;
                margin:5px 10px;
            }
            .color_circle{
                width:20px;
                height:20px;
                border-radius:20px;
                position:relative;
                left:10px
            }
        </style>
        """
        #creamos la leyenda con un bucle y html
        html_legend="<div class='legend_container'>"
        for key,val in VIRIDIS_6.items():
            html_legend+=f"<div class='color_container'><label>Grupo {key}:</label><div class='color_circle' style='background-color:{val}'></div></div>"
        
        html_legend+=f"<div class='color_container'><label>Prediccion:</label><div class='color_circle' style='background-color:red'></div></div>"
        html_legend+="</div>"
        st.markdown(html_legend_style + html_legend,unsafe_allow_html=True)
    except:
        st.warning("No se pudo cargar el mapa, falta el csv en ../data/raw/")

    st.write("---")
    st.write("Recursos usados: streamlit, pandas, sklearn y pydeck.")

elif opcion == "Ver Notebook":
    st.title("📓 Visualizacion del Notebook")
    st.write("Este es el trabajo previo de analisis que hice en el Jupiter.")
    
    # leer el html generado con "jupyter nbconvert --to html --template lab --theme dark notebooks/09-k-means-housing.ipynb"
  
    if os.path.exists(JUPITER_PATH):
        with open(JUPITER_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # insertar el html en un componente
        components.html(html_content, height=800, scrolling=True)
    else:
        st.error("No se encontro el archivo HTML del notebook")
