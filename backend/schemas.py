from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    username: str
    password: str

class EmotionLog(BaseModel):
    session_id: int
    emotion: str
    confidence: float

class SessionLog(BaseModel):
    user_id: int
    conversation: str
