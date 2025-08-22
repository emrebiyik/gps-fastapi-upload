from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class GPSData(Base):
    __tablename__ = "gps_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    filename = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    flag = Column(String, nullable=True)   # "normal" or "abnormal"
    taken_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# optional: explicit exports
__all__ = ["GPSData"]