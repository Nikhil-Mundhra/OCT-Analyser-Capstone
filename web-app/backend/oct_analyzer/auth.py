from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic_settings import BaseSettings

class AuthSettings(BaseSettings):
    api_key: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = AuthSettings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validates the X-API-Key header against the configured API key.
    If no API key is set in settings, auth is bypassed for development.
    """
    if not settings.api_key:
        return None
        
    if api_key_header == settings.api_key:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )
