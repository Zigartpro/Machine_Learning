import pandas as pd
import numpy as np

def similitud_coseno(a, b):
    # Devuelve la similitud coseno entre dos vectores
    numerador = np.dot(a, b)
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador == 0:
        return 0
    return numerador / denominador

def distancia_euclidiana(a, b):
    # Devuelve la distancia euclidiana entre dos vectores
    return np.linalg.norm(a - b)

class KNNManual:
    def __init__(self, k=5, metrica="coseno"):
        self.k = k
        self.metrica = metrica
        self.X_entrenamiento = None
        self.y_entrenamiento = None

    def entrenar(self, X, y):
        # Guarda los datos de entrenamiento
        self.X_entrenamiento = X
        self.y_entrenamiento = np.array(y)

    def _calcular_similitud(self, x):
        # Calcula distancias o similitudes respecto al vector x
        if self.metrica == "coseno":
            puntajes = np.array([similitud_coseno(x, fila) for fila in self.X_entrenamiento])
            return puntajes, True
        else:
            distancias = np.array([distancia_euclidiana(x, fila) for fila in self.X_entrenamiento])
            return distancias, False

    def predecir(self, x):
        # Predice la clase de usuario para el vector x
        puntajes, es_similitud = self._calcular_similitud(x)

        if es_similitud:
            indices = np.argsort(-puntajes)[:self.k]
            vecinos = self.y_entrenamiento[indices]
            pesos = puntajes[indices]
        else:
            indices = np.argsort(puntajes)[:self.k]
            vecinos = self.y_entrenamiento[indices]
            pesos = 1 / (puntajes[indices] + 1e-9)

        votos = {}
        for etiqueta, peso in zip(vecinos, pesos):
            votos[etiqueta] = votos.get(etiqueta, 0) + peso

        return max(votos, key=votos.get)

def recomendar_canciones(modelo, vector_usuario, df_canciones, columnas_canciones, top_n=10):
    # Genera recomendaciones para un nuevo usuario
    vecinos = modelo.X_entrenamiento
    similitudes = np.array([similitud_coseno(vector_usuario, v) for v in vecinos])
    indices_top = np.argsort(-similitudes)[:modelo.k]
    pesos = similitudes[indices_top]

    canciones_vecinos = df_canciones.iloc[indices_top].values
    puntuaciones = np.dot(pesos, canciones_vecinos) / (np.sum(pesos) + 1e-9)

    no_puntuadas = vector_usuario == 0
    puntuaciones = np.where(no_puntuadas, puntuaciones, -np.inf)

    top_indices = np.argsort(-puntuaciones)[:top_n]
    recomendaciones = [(columnas_canciones[i], float(puntuaciones[i])) for i in top_indices if puntuaciones[i] > 0]
    return recomendaciones

# Géneros musicales
GENEROS = {
    "Vallenato": ["La Piragua","La Gota Fria","Fruta Fresca","Alicia Adorada","Oye Bonita","Matilde Lina",
                  "La Celosa","El Santo Cachon","Tierra Mala","La Ventana Marroncita","Mi Hermano y Yo",
                  "La Canaguatera","Volvi a Nacer","Dejala","El Cantor de Fonseca","Amor de Primavera",
                  "El Amor Mas Grande del Planeta","Obsesion","La Casa en el Aire","El Testamento",
                  "Tu Numero Equivocado","La Duena de mi Canto","La Mujer del Pelotero","La Espinita",
                  "Dime Pajarito","Los Caminos de la Vida","La Comadre","Parranda en el Cafetal",
                  "El Cantinero","Ausencia","Por un Amor","Dame un Besito","Solo para Ti"],
    
    "Folclor": ["Colombia Tierra Querida","La Pollera Colora","Cumbia Cienaguera","El Pescador","Cumbia Sampuesana",
                "Soy Colombiano","Cumbia del Rio","La Mucura","El Porro Sabanero","Mi Buenaventura","Carmen de Bolivar",
                "El Sanjuanero","El Toro","Prende la Vela","Tambo Tambo","El Currulao","La Cumbia Alegre","Tierra del Sol",
                "La Nina Lola","Cumbia del Caribe","Porro Bonito","La Aventurera","El Bocachico","Las Tapas","El Carnaval",
                "San Pedro en el Espinal","La Danza del Garabato","El Gallo Tuerto","Cumbia del Magdalena","Cumbia Bonita",
                "Fandanguillo","La Cumbia Cienaguera Remix","El Merecumbe"],
    
    "Rock": ["Bolero Falaz","Florecita Rockera","De Musica Ligera","Lamento Boliviano","Persiana Americana",
             "Rayando el Sol","Clavado en un Bar","La Tierra","Eres","En el Muelle de San Blas","Me Gustas Tu",
             "Labios Compartidos","Ciudad de Luz","Suenos Rotos","Mi Generacion","A Contraluz","Cuando Pase el Temblor",
             "Amor Eterno","Matame","Universo","Salir Corriendo","Dia Cero","El Sonido del Silencio","Trapos",
             "Sin Documentos","Cancion para un Final","Luz de Dia","El Espacio Interior","Nada Personal","Vivo en Mi",
             "La Ventana Azul","Tu Mirada","Ecos de Amor"],
    
    "Pop": ["La Camisa Negra","A Dios le Pido","Es Por Ti","Fotografia","Volverte a Ver","Color Esperanza",
            "Robarte un Beso","La Playa","Por Amarte Asi","Todo Cambio","Bendita tu Luz","Hoy","Decidiste Dejarme",
            "Tal Vez","Mientes","Te Espere","Algo Mas","Entra en mi Vida","Quisiera","Besos Usados",
            "El Amor Despues del Amor","Corre","Tu Amor","Limon y Sal","Eres para Mi","El Amor de Mi Vida",
            "Quedate Conmigo","Cada Dia","Te Mando Flores","Me Enamora","Te Amo","No Creo","Perfecto para Mi"],
    
    "Urbano": ["Tusa","Hawai","Mi Gente","Ginza","Felices los 4","6 AM","Chantaje","Ay Dios Mio","Que Pretendes",
               "Sobrio","Borro Cassette","Sin Contrato","Dont Be Shy","Poblado","Criminal","Mamiii","Cuatro Babys",
               "Despacito","Pepas","Vente Pa Ca","Ojitos Lindos","Me Porto Bonito","Amorfoda","Hey Mor","Callaita",
               "Te Bote","Safari","X Equis","Provenza Remix","Coco Chanel","La Cancion","Provenza","Sin Medir Distancias"],
    
    "Salsa": ["Cali Pachanguero","Rebelion","Idilio","Gitana","Juanito Alimana","Devorame Otra Vez","Lloraras",
              "Mi Gente Salsa","Periodico de Ayer","El Gran Varon","Brujeria","Conteo Regresivo","Tu con El",
              "Plante Bandera","Quitate Tu","La Cartera","El Cantante","Todo Tiene su Final","Me Libere",
              "Salsa y Control","Anacaona","Una Aventura","Oiga Mire Vea","Que Precio Tiene el Cielo",
              "Casi Te Envidio","Nadie Como Tu","Ven Devorame Otra Vez","Quiero Dormir Cansado","Dejate Querer",
              "Cali Aji","Gitana Salsa","Idilio de Amor","El Amor de la Salsa","Te Conozco Bien","Cali Vive",
              "Te Ensenare a Olvidar"]
}

def get_generos_canciones():
    # Retorna diccionario canción → género
    generos_canciones = {}
    for genero, canciones in GENEROS.items():
        for cancion in canciones:
            generos_canciones[cancion] = genero
    return generos_canciones

class MusicRecommenderSystem:
    def __init__(self, csv_path, k=5, metrica="coseno"):
        self.k = k
        self.metrica = metrica
        self.modelo = None
        self.columnas_canciones = None
        self.df_entrenamiento = None
        self.generos_canciones = get_generos_canciones()
        self.cargar_dataset(csv_path)
        
    def cargar_dataset(self, csv_path):
        # Carga y prepara el dataset
        df = pd.read_csv(csv_path, encoding='utf-8')
        columnas_meta = ["UserID", "Edad", "Genero", "Region", "GeneroFav", "ClaseUsuario"]
        self.columnas_canciones = [c for c in df.columns if c not in columnas_meta]
        
        X = df[self.columnas_canciones].fillna(0).values
        y = df["ClaseUsuario"].values
        
        # División entrenamiento/prueba
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_prueba = int(n * 0.2)
        X_entrenamiento = X[indices[n_prueba:]]
        y_entrenamiento = y[indices[n_prueba:]]
        
        self.modelo = KNNManual(k=self.k, metrica=self.metrica)
        self.modelo.entrenar(X_entrenamiento, y_entrenamiento)
        self.df_entrenamiento = pd.DataFrame(X_entrenamiento, columns=self.columnas_canciones)
        
    def obtener_canciones_por_genero(self):
        # Retorna las canciones organizadas por género
        return GENEROS
    
    def predecir_y_recomendar(self, ratings, top_n=10):
        # Predice clase y genera recomendaciones
        candidato = np.zeros(len(self.columnas_canciones))
        
        for cancion, rating in ratings.items():
            if cancion in self.columnas_canciones:
                idx = self.columnas_canciones.index(cancion)
                candidato[idx] = rating
        
        clase_predicha = self.modelo.predecir(candidato)
        recomendaciones = recomendar_canciones(
            self.modelo, 
            candidato, 
            self.df_entrenamiento, 
            self.columnas_canciones, 
            top_n=top_n
        )
        
        # Agregar género a las recomendaciones
        recomendaciones_con_genero = []
        for cancion, puntaje in recomendaciones:
            genero = self.generos_canciones.get(cancion, "Desconocido")
            recomendaciones_con_genero.append({
                "cancion": cancion,
                "puntaje": puntaje,
                "genero": genero
            })
        
        return {
            "clase_predicha": clase_predicha,
            "recomendaciones": recomendaciones_con_genero
        }