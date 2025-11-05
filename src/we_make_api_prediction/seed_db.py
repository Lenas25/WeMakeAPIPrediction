import firebase_admin
from firebase_admin import credentials, firestore
import random
from datetime import datetime, timedelta

# --- 1. CONFIGURACIÓN ---

# ¡¡¡CAMBIA ESTO!!! Pon el ID del tablero donde quieres crear las tareas de prueba.
BOARD_ID_PARA_PRUEBAS = "yhaq6Jvq8coi2oWUMy78" # <-- REEMPLAZA ESTO

# Conecta a Firebase (asegúrate de que tu 'serviceAccountKey.json' esté en la misma carpeta)
cred = credentials.Certificate("src/we_make_api_prediction/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

print("Conectado a Firebase. Preparando para generar datos...")

# --- 2. DATOS DE EJEMPLO ---

# Lista de usuarios (miembros y revisores). Reemplaza con IDs reales de tu base de datos.
# Puedes obtenerlos de la sección 'Authentication' de tu consola de Firebase.
user_ids = [
    "FsJXZi59FKVV8ftaT8NHWqg9jvB2",
    "1bIK7cV6QuQHFhf6G6Mt3MCe9e83",
]

# Títulos y descripciones de ejemplo para las tareas
task_titles = [
    "Diseñar el nuevo mockup para la pantalla de inicio",
    "Implementar el login con Google", "Corregir el bug en el perfil de usuario",
    "Desarrollar la funcionalidad de chat", "Escribir la documentación de la API",
    "Preparar la presentación para la reunión de sprint", "Optimizar la carga de imágenes",
    "Crear los tests unitarios para el módulo de pagos", "Investigar nueva librería de gráficos",
    "Actualizar las dependencias del proyecto", "Refactorizar el servicio de notificaciones",
]

# --- 3. FUNCIÓN PARA CREAR UNA TAREA FICTICIA ---

def create_dummy_task(board_id, all_users):
    """
    Crea un diccionario que representa un documento de tarea con datos aleatorios pero realistas.
    """
    # Fechas
    days_ago_created = random.randint(10, 90) # Creada entre 10 y 90 días atrás
    created_at = datetime.now() - timedelta(days=days_ago_created)
    
    task_duration_days = random.randint(3, 15) # La tarea debería durar entre 3 y 15 días
    deadline = created_at + timedelta(days=task_duration_days)
    
    # Simular si la tarea se completó a tiempo o con retraso
    is_completed = random.choice([True, True, False]) # 2/3 de probabilidad de estar completada
    approved_at = None
    status = random.choice(["pending", "in_progress"])
    
    if is_completed:
        status = "completed"
        # ¿Se completó con retraso? 30% de probabilidad
        is_late = random.random() < 0.30
        if is_late:
            # Se completó unos días después de la fecha límite
            days_late = random.randint(1, 5)
            approved_at = deadline + timedelta(days=days_late)
        else:
            # Se completó unos días antes de la fecha límite
            days_early = random.randint(1, task_duration_days - 1)
            approved_at = deadline - timedelta(days=days_early)

    # Miembros y Revisor
    num_assigned = random.randint(1, 3)
    num_assigned = min(max(num_assigned, 0), len(all_users))
    assigned_members = random.sample(all_users, num_assigned) if num_assigned > 0 else []
    
    # Asegurarse de que el revisor no esté en la lista de asignados
    possible_reviewers = [uid for uid in all_users if uid not in assigned_members]
    reviewer_id = random.choice(possible_reviewers) if possible_reviewers else all_users[0]
    
    # Subtareas
    num_subtasks = random.randint(0, 5)
    subtasks = [{"text": f"Subtarea de ejemplo {i+1}", "completed": random.choice([True, False])} for i in range(num_subtasks)]
    
    # Crear el objeto final de la tarea
    task = {
        "boardId": board_id,
        "title": random.choice(task_titles),
        "description": "Esta es una descripción generada automáticamente para la tarea de prueba.",
        "priority": random.choice(["low", "medium", "high"]),
        "status": status,
        "createdAt": created_at,
        "deadline": deadline,
        "approvedAt": approved_at,
        "createdBy": random.choice(all_users),
        "approvedBy": random.choice(all_users) if is_completed else None,
        "assignedMembers": assigned_members,
        "reviewerId": reviewer_id,
        "subtasks": subtasks,
        "rewardPoints": random.choice([10, 20, 30, 50]),
        "penaltyPoints": random.choice([5, 10, 15]),
        "penaltyApplied": False, # Siempre empezamos con la penalidad no aplicada
        "completedAt": approved_at if is_completed else None,
    }
    return task

# --- 4. EJECUTAR EL SCRIPT ---

def seed_tasks(num_tasks=50):
    """
    Función principal que genera y sube un número_de_tareas a Firestore.
    """
    if not BOARD_ID_PARA_PRUEBAS or BOARD_ID_PARA_PRUEBAS == "TU_BOARD_ID_AQUI":
        print("\nERROR: ¡Debes editar el script y poner un BOARD_ID válido!\n")
        return

    tasks_collection_ref = db.collection("tasks")
    
    print(f"Generando {num_tasks} tareas para el tablero '{BOARD_ID_PARA_PRUEBAS}'...")
    
    for i in range(num_tasks):
        dummy_task = create_dummy_task(BOARD_ID_PARA_PRUEBAS, user_ids)
        
        tasks_collection_ref.add(dummy_task)
        print(f"  -> Tarea {i + 1}/{num_tasks} creada.")
        
    print(f"\n¡Proceso completado! Se han añadido {num_tasks} tareas de prueba a tu base de datos.")

if __name__ == "__main__":
    seed_tasks()