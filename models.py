from pydantic import BaseModel
from typing import Optional

class ImageMetadata(BaseModel):
    category: Optional[str]
    name: Optional[str]
    address: Optional[str]