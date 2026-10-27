from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field
import uuid
import time

AssetCategory = Literal["robot", "obstacle", "sensor", "template"]

class AssetMetadata(BaseModel):
    id: str = Field(default_factory=lambda: f"asset_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., min_length=1, max_length=100)
    category: AssetCategory
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    is_built_in: bool = False
    created_at: float = Field(default_factory=time.time)
    thumbnail_hint: str = "box" # "bot", "arm", "box", "cylinder", "sphere", "radio", "video", "template", etc.
    version: str = "1.0.0"

class AssetModel(AssetMetadata):
    data: Dict[str, Any] = Field(default_factory=dict)

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Optional[Dict[str, Any]] = None

class CreateAssetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    category: AssetCategory
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    data: Dict[str, Any]
    thumbnail_hint: Optional[str] = None

class ApplyAssetRequest(BaseModel):
    position_offset: Optional[List[float]] = None
    replace_robot: bool = True
