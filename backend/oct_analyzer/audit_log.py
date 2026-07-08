import logging
import json
from datetime import datetime
from typing import Any

# Configure a specific logger for audit events
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# Optional: write audit logs to a specific file
handler = logging.FileHandler("audit.log")
formatter = logging.Formatter('%(asctime)s - AUDIT - %(message)s')
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

def log_audit_event(event_type: str, user_id: str, action: str, resource_id: str, details: dict[str, Any] = None):
    """
    Records an enterprise audit event.
    """
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "user_id": user_id,
        "action": action,
        "resource_id": resource_id,
        "details": details or {}
    }
    audit_logger.info(json.dumps(event))

def log_scan_accessed(scan_id: str, user_id: str = "system"):
    log_audit_event("scan_access", user_id, "read", scan_id)

def log_scan_created(scan_id: str, user_id: str = "system"):
    log_audit_event("scan_create", user_id, "create", scan_id)
