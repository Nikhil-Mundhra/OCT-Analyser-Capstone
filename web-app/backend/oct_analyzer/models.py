from sqlalchemy import Column, String, Boolean, Text, JSON
from .database import Base
import uuid

class ScanRecord(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, index=True, default=lambda: uuid.uuid4().hex)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    task_id = Column(String, nullable=True)     # Celery task ID
    detail = Column(Text, nullable=True)        # For error messages
    result = Column(JSON, nullable=True)        # To store the final segmentation/classification payload
    is_demo_model = Column(Boolean, default=False)
