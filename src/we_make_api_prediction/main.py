# =============================================================================
#   Paso 1: Importar las herramientas que necesitamos
# =============================================================================
import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
import os

# Herramientas para conectar a Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Herramientas para organizar y analizar datos
import pandas as pd
import numpy as np

# Herramientas para construir Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Para guardar y cargar una vez que ha aprendido

# --- Configuración Inicial ---

# Creamos la aplicación de API
app = FastAPI()

# Nos conectamos a Firebase.
cred = credentials.Certificate("src/we_make_api_prediction/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# =============================================================================
#   Paso 2: Construir
# =============================================================================

# --- Funciones de Machine Learning ---

def prepare_features_for_model(df):
    """
    Esta función es como un traductor.
    Solo entiende números. Esta función traduce las palabras a números.
    """
    df_features = df.copy()

    # 1. Traducir Prioridad: "low" -> 0, "medium" -> 1, "high" -> 2
    priority_mapping = {'low': 0, 'medium': 1, 'high': 2}
    # Si no tiene prioridad, asumimos "medium" (1)
    df_features['priority_encoded'] = df_features['priority'].map(priority_mapping).fillna(1)

    # 2. Calcular Días para la Fecha Límite:
    # ¿Cuántos días faltan para la fecha límite? Si es en el futuro, es un número positivo.
    # Si ya pasó, es un número negativo.
    # Usamos .tz_localize(None) para poder comparar las fechas sin problemas de zona horaria.
    df_features['days_to_deadline'] = (df_features['deadline'].dt.tz_localize(None) - datetime.now()).dt.days
    df_features['days_to_deadline'] = df_features['days_to_deadline'].fillna(30)

    # 3. Contar Subtareas: ¿Cuántas subtareas tiene? Más subtareas = más complejo.
    df_features['subtasks_count'] = df_features['subtasks'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df_features


def train_risk_prediction_model(df_history):
    """
    Esta es la función que "enseña".
    Le mostramos todas las tareas del pasado y le decimos cuáles se retrasaron.
    Aprende los patrones y se guarda a sí misma para no tener que aprender de nuevo cada vez.
    """
    MODEL_FILE_PATH = "models/risk_model.pkl"

    # Si ya hemos enseñado a la Bola de Cristal antes, simplemente la usamos.
    if os.path.exists(MODEL_FILE_PATH):
        try:
            print("Cargando modelo de riesgo existente...")
            return joblib.load(MODEL_FILE_PATH)
        except Exception as e:
            print(f"No se pudo cargar el modelo, se re-entrenará. Error: {e}")

    print("Entrenando un nuevo modelo de riesgo...")
    df_train = df_history.copy()
    # Solo podemos aprender de tareas que ya terminaron y tenían una fecha límite.
    df_train = df_train[df_train['status'] == 'completed'].copy()
    if 'completedAt' in df_train.columns:
        df_train = df_train.dropna(subset=['completedAt', 'deadline'])
        # Traducimos los datos a números que el modelo entienda.
        df_train = prepare_features_for_model(df_train)
        features_to_use = ['priority_encoded', 'days_to_deadline', 'subtasks_count']
        X = df_train[features_to_use]
        # La "respuesta correcta": ¿La tarea se retrasó (1) o no (0)?
        # Usamos .tz_localize(None) para comparar sin problemas de zona horaria.
        df_train['is_late'] = (df_train['completedAt'].dt.tz_localize(None) > df_train['deadline'].dt.tz_localize(None)).astype(int)
        y = df_train['is_late']
        if len(df_train) < 10 or len(y.unique()) < 2:
            print("No hay suficientes datos para entrenar un modelo significativo.")
            class DummyModel: # type: ignore
                def predict(self, X): return np.zeros(len(X))
            return DummyModel()
    else:
        print("No hay campo 'completedAt' en los datos de entrenamiento.")
        class DummyModel:
            def predict(self, X): return np.zeros(len(X))
        return DummyModel()

    # Dividimos: una parte para enseñar, otra para hacer un examen.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos un clasificador de Bosque Aleatorio algoritmo que te ayuda a tomar decisiones basadas en datos
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Le hacemos el examen para ver qué tan bien aprendió.
    predictions = model.predict(X_test)
    print(f"Precisión del nuevo modelo de riesgo: {accuracy_score(y_test, predictions) * 100:.2f}%")

    # Guardamos ya entrenada para la próxima vez.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

    return model


# =============================================================================
#   Paso 3: Abrir el Endpoint de la API
# =============================================================================

@app.get("/dashboard/{board_id}")
async def get_dashboard_data(board_id: str):
    """
    Esta es la función principal que tu app de Android llamará.
    Cuando la app pide datos para un tablero, esta función se ejecuta.
    """
    try:
        # --- Obtener los Datos de Firebase ---
        tasks_ref = db.collection('tasks').where('boardId', '==', board_id)
        docs = tasks_ref.stream()
        tasks_list = [doc.to_dict() for doc in docs]
        
        if not tasks_list:
            # Si no hay tareas, no podemos calcular nada.
            return {"message": "No hay tareas en este tablero para analizar."}

        # --- Usar Pandas, nuestra "Hoja de Cálculo" ---
        df = pd.DataFrame(tasks_list)
        
        # Corregir los tipos de datos de fecha
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
        # Solo convertir completedAt si existe y si la tarea está completada
        if 'completedAt' in df.columns:
            df.loc[df['status'] == 'completed', 'completedAt'] = pd.to_datetime(df.loc[df['status'] == 'completed', 'completedAt'], errors='coerce')
        df = df.dropna(subset=['createdAt']) # Ignorar tareas con fecha de creación corrupta

        # --- Parte 1: Usar la "Máquina de Contar" (Métricas) ---
        
        # Tareas completadas en las últimas 4 semanas
        df_completed = df[df['status'] == 'completed'].copy()
        
        if not df_completed.empty and 'completedAt' in df_completed.columns:
            df_completed = df_completed.dropna(subset=['completedAt', 'deadline'])
            # Creamos una clave única para cada semana, ej: (2025, 42)
            df_completed['year_week'] = df_completed['completedAt'].dt.isocalendar().apply(
                lambda x: (x.year, x.week), axis=1
            )
            tasks_per_week_series = df_completed.groupby('year_week').size().tail(4)
            
            # Formateamos las claves para el JSON, ej: (2025, 42) -> "S42"
            tasks_per_week = {}
            for idx, count in tasks_per_week_series.items():
                # idx puede ser una tupla (year, week) o algún otro tipo hashable; manejamos ambos casos
                if isinstance(idx, (list, tuple)) and len(idx) >= 2:
                    week = idx[1]
                else:
                    # intentar usar atributo 'week' si existe, sino usar la representación de cadena
                    week = getattr(idx, 'week', None)
                    if week is None:
                        week = str(idx)
                tasks_per_week[f"S{week}"] = int(count)

            # Calcular la tasa de finalización a tiempo (porcentaje)
            df_completed.loc[:, 'on_time'] = (df_completed['completedAt'].dt.tz_localize(None) <= df_completed['deadline'].dt.tz_localize(None))
            on_time_rate = df_completed['on_time'].mean() * 100 if not df_completed.empty else 100
        else:
            tasks_per_week = {}
            on_time_rate = 100
        # Distribución de tareas por prioridad
        priority_distribution = df['priority'].value_counts().to_dict()

        # --- Parte 2: Usar Predicciones ---
        
        # Primero, le enseñamos con el historial de tareas
        risk_model = train_risk_prediction_model(df)
        
        # Ahora, le mostramos las tareas que aún no están terminadas para que adivine.
        df_pending = df[df['status'] != 'completed'].copy()
        at_risk_tasks = []
        if not df_pending.empty:
            df_predict = prepare_features_for_model(df_pending)
            features_to_use = ['priority_encoded', 'days_to_deadline', 'subtasks_count']
            
            predictions = risk_model.predict(df_predict[features_to_use])
            df_pending['risk_prediction'] = predictions
            
            # De la lista de tareas pendientes, nos quedamos con las que la Bola de Cristal dijo que "se romperán" (riesgo = 1)
            at_risk_tasks = df_pending[df_pending['risk_prediction'] == 1][['title', 'priority']].head(3).to_dict('records')

        # --- Parte 3: Juntar todo en un solo paquete (JSON) para enviarlo a la app de Android ---
        
        dashboard_json = {
            "summary": {
                "total_tasks": len(df),
                "pending_tasks": len(df[df['status'] != 'completed']),
            },
            "productivity": {
                # Convertimos las claves de fecha a texto para que el JSON sea válido
               "tasks_completed_per_week": tasks_per_week,
                "priority_distribution": {str(k): int(v) for k, v in priority_distribution.items()},
                "on_time_completion_rate": round(on_time_rate, 2)
            },
            "predictions": {
                "at_risk_tasks": at_risk_tasks
            }
        }
        
        return dashboard_json

    except Exception as e:
        print(f"Ocurrió un error: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")

# =============================================================================
#   Paso 4: Iniciar el Servidor
#           En esta aplicacion se usa poetry para manejar las dependencias
#           Asegurate de tener poetry instalado y correr el siguiente comando en la terminal:
#           poetry install
#           Luego, para iniciar el servidor, corre:
#           poetry run uvicorn we_make_api_prediction
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)