# =============================================================================
#   Paso 1: Importar las herramientas que necesitamos
# =============================================================================
import os
from datetime import datetime

# Herramientas para conectar a Firebase
import firebase_admin
import joblib  # Para guardar y cargar una vez que ha aprendido
import numpy as np
# Herramientas para organizar y analizar datos
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Herramientas para construir Machine Learning
from sklearn.model_selection import train_test_split

# --- Configuración Inicial ---

# Creamos la aplicación de API
app = FastAPI()

carpeta_actual = os.path.dirname(os.path.abspath(__file__))

SERVICE_ACCOUNT_KEY_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")

if SERVICE_ACCOUNT_KEY_JSON:
    # Si la variable de entorno existe (estamos en Render)
    import json

    # La credencial espera un diccionario, por lo que cargamos la cadena JSON
    service_account_info = json.loads(SERVICE_ACCOUNT_KEY_JSON)
    cred = credentials.Certificate(service_account_info)
    print("Firebase inicializado usando la variable de entorno.")
else:
    # Si la variable no existe (usamos el archivo local para desarrollo)
    # Importamos solo si es necesario para evitar problemas de dependencias en Render.
    from firebase_admin import credentials
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    # Asume que tu archivo está en la raíz de /src/ o directamente en src/we_make_api_prediction/
    # Según tu captura de pantalla, está en el mismo directorio que main.py
    ruta_json = os.path.join(carpeta_actual, "serviceAccountKey.json")
    cred = credentials.Certificate(ruta_json)
    print("Firebase inicializado usando el archivo local serviceAccountKey.json.")

# Inicializar Firebase
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
    df_features['priority_encoded'] = df_features['priority'].map(
        priority_mapping).fillna(1)

    # 2. Calcular Días para la Fecha Límite:
    # ¿Cuántos días faltan para la fecha límite? Si es en el futuro, es un número positivo.
    # Si ya pasó, es un número negativo.
    # Usamos .tz_localize(None) para poder comparar las fechas sin problemas de zona horaria.
    df_features['days_to_deadline'] = (
        df_features['deadline'].dt.tz_localize(None) - datetime.now()).dt.days
    df_features['days_to_deadline'] = df_features['days_to_deadline'].fillna(
        30)

    # 3. Contar Subtareas: ¿Cuántas subtareas tiene? Más subtareas = más complejo.
    df_features['subtasks_count'] = df_features['subtasks'].apply(
        lambda x: len(x) if isinstance(x, list) else 0)

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
        features_to_use = ['priority_encoded',
                           'days_to_deadline', 'subtasks_count']
        X = df_train[features_to_use]
        # La "respuesta correcta": ¿La tarea se retrasó (1) o no (0)?
        # Usamos .tz_localize(None) para comparar sin problemas de zona horaria.
        df_train['is_late'] = (df_train['completedAt'].dt.tz_localize(
            None) > df_train['deadline'].dt.tz_localize(None)).astype(int)
        y = df_train['is_late']
        if len(df_train) < 10 or len(y.unique()) < 2:
            print("No hay suficientes datos para entrenar un modelo significativo.")

            class DummyModel:  # type: ignore
                def predict(self, X): return np.zeros(len(X))
            return DummyModel()
    else:
        print("No hay campo 'completedAt' en los datos de entrenamiento.")

        class DummyModel:
            def predict(self, X): return np.zeros(len(X))
        return DummyModel()

    # Dividimos: una parte para enseñar, otra para hacer un examen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Creamos un clasificador de Bosque Aleatorio algoritmo que te ayuda a tomar decisiones basadas en datos
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Le hacemos el examen para ver qué tan bien aprendió.
    predictions = model.predict(X_test)
    print(
        f"Precisión del nuevo modelo de riesgo: {accuracy_score(y_test, predictions) * 100:.2f}%")

    # Guardamos ya entrenada para la próxima vez.
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)

    return model


# =============================================================================
#   Paso 3: Abrir el Endpoint de la API
# =============================================================================

# --- Endpoint 1: Dashboard General (Tu código existente, sin cambios) ---
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
            df.loc[df['status'] == 'completed', 'completedAt'] = pd.to_datetime(
                df.loc[df['status'] == 'completed', 'completedAt'], errors='coerce')
        # Ignorar tareas con fecha de creación corrupta
        df = df.dropna(subset=['createdAt'])

        # --- Parte 1: Usar la "Máquina de Contar" (Métricas) ---

        # Tareas completadas en las últimas 4 semanas
        df_completed = df[df['status'] == 'completed'].copy()

        if not df_completed.empty and 'completedAt' in df_completed.columns:
            df_completed = df_completed.dropna(
                subset=['completedAt', 'deadline'])
            # Creamos una clave única para cada semana, ej: (2025, 42)
            df_completed['year_week'] = df_completed['completedAt'].dt.isocalendar().apply(
                lambda x: (x.year, x.week), axis=1
            )
            tasks_per_week_series = df_completed.groupby(
                'year_week').size().tail(4)

            tasks_per_week = {}
            for idx, count in tasks_per_week_series.items():
                # idx puede ser una tupla (year, week) o algún otro tipo hashable; manejamos ambos casos
                if isinstance(idx, (list, tuple)) and len(idx) >= 2:
                    week = idx[1]
                else:
                    week = getattr(idx, 'week', None)
                    if week is None:
                        week = str(idx)
                tasks_per_week[f"S{week}"] = int(count)

            # Calcular la tasa de finalización a tiempo (porcentaje)
            df_completed.loc[:, 'on_time'] = (df_completed['completedAt'].dt.tz_localize(
                None) <= df_completed['deadline'].dt.tz_localize(None))
            on_time_rate = df_completed['on_time'].mean(
            ) * 100 if not df_completed.empty else 100
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
            features_to_use = ['priority_encoded',
                               'days_to_deadline', 'subtasks_count']

            predictions = risk_model.predict(df_predict[features_to_use])
            df_pending['risk_prediction'] = predictions

            # De la lista de tareas pendientes, nos quedamos con las que la Bola de Cristal dijo que "se romperán" (riesgo = 1)
            at_risk_tasks = df_pending[df_pending['risk_prediction'] == 1][[
                'title', 'priority']].head(3).to_dict('records')

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
        raise HTTPException(
            status_code=500, detail=f"Ocurrió un error interno: {e}")

# --- Endpoint 2: - Resumen de un Usuario Específico ---


@app.get("/summary/{board_id}/{user_id}")
async def get_user_summary_in_board(board_id: str, user_id: str):
    """
    Calcula y devuelve las estadísticas de un solo usuario dentro de un tablero.
    """
    try:
        # 1. Obtener todas las tareas del tablero
        tasks_ref = db.collection('tasks').where('boardId', '==', board_id)
        docs = tasks_ref.stream()
        tasks_list = [doc.to_dict() for doc in docs]

        if not tasks_list:
            return {"message": "No hay tareas en este tablero para analizar."}

        df = pd.DataFrame(tasks_list)
        # Asegurar que las columnas existen antes de convertir
        df['completedAt'] = pd.to_datetime(
            df['completedAt'], errors='coerce') if 'completedAt' in df.columns else pd.NaT
        df['deadline'] = pd.to_datetime(
            df['deadline'], errors='coerce') if 'deadline' in df.columns else pd.NaT

        # 2. Filtrar las tareas que pertenecen a ESTE usuario
        # Una tarea le "pertenece" si es miembro asignado O es el revisor.
        df_user = df[
            (df['assignedMembers'].apply(lambda members: user_id in members if isinstance(members, list) else False)) |
            (df['reviewerId'] == user_id)
        ].copy()

        if df_user.empty:
            return {
                "user_id": user_id,
                "tasks_involved": 0,
                "tasks_completed": 0,
                "on_time_rate": 100.0,
            }

        # 3. Calcular las métricas personales
        tasks_involved = len(df_user)
        df_user_completed = df_user[df_user['status'] == 'completed']
        tasks_completed = len(df_user_completed)

        now = datetime.now()
        overdue_tasks = len(df_user[
            (df_user['deadline'].dt.tz_localize(None) < now) &
            (df_user['status'] != 'completed')
        ])

        # Tasa de cumplimiento personal
        # Hacemos una comparación segura por fila convirtiendo a timestamps numéricos
        on_time_tasks = 0
        for _, row in df_user_completed.iterrows():
            # Convert entries to pandas Timestamp safely; pd.Timestamp handles many input types.
            try:
                comp_val = row.get('completedAt', None)
                comp_ts = pd.Timestamp(
                    comp_val) if comp_val is not None else pd.NaT
            except Exception:
                comp_ts = pd.NaT

            try:
                dl_val = row.get('deadline', None)
                dl_ts = pd.Timestamp(dl_val) if dl_val is not None else pd.NaT
            except Exception:
                dl_ts = pd.NaT

            if pd.isna(comp_ts) or pd.isna(dl_ts):
                continue
            try:
                # Usar .value (nanoseconds since epoch) evita problemas de zona horaria al comparar
                if int(comp_ts.value) <= int(dl_ts.value):
                    on_time_tasks += 1
            except Exception:
                # En caso de cualquier problema al convertir, ignoramos esa fila
                continue
        on_time_rate = (on_time_tasks / tasks_completed) * \
            100 if tasks_completed > 0 else 100

        return {
            "user_id": user_id,
            "tasks_involved": tasks_involved,
            "tasks_completed": tasks_completed,
            "overdue_tasks": overdue_tasks,
            "on_time_rate": round(on_time_rate, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error: {e}")


# --- Endpoint 3: - Leaderboard (Clasificación de Miembros) ---
@app.get("/leaderboard/{board_id}")
async def get_leaderboard(board_id: str):
    """
    Devuelve una lista de miembros de un tablero ordenados por sus puntos.
    """

    try:
        members_details_ref = db.collection("boards").document(
            board_id).collection("members_details")
        members_docs = members_details_ref.stream()

        member_points_list = []
        user_ids_to_fetch = []
        for doc in members_docs:
            member_data = doc.to_dict()
            member_points_list.append({
                "user_id": doc.id,
                "points": member_data.get("points", 0)
            })
            user_ids_to_fetch.append(doc.id)

        if not user_ids_to_fetch:
            return []  # Devolver una lista vacía si no hay miembros

        users_ref = db.collection("users").where(
            "__name__", "in", user_ids_to_fetch)
        user_docs = users_ref.stream()

        user_info_map = {doc.id: doc.to_dict() for doc in user_docs}

        leaderboard = []
        for member in member_points_list:
            user_id = member["user_id"]
            user_data = user_info_map.get(user_id)
            if user_data:
                leaderboard.append({
                    "name": user_data.get("name", "Usuario Desconocido"),
                    "photoUrl": user_data.get("photoUrl", ""),
                    "points": member["points"]
                })

        leaderboard.sort(key=lambda x: x["points"], reverse=True)

        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error: {e}")

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
