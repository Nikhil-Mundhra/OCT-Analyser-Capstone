import os
from celery import Celery

# Configure Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "oct_analyzer_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
