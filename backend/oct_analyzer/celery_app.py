from celery import Celery
from .constants import REDIS_URL

celery_app = Celery(
    "oct_analyzer_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.oct_analyzer.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
