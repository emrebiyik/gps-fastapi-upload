from sqlalchemy import Column, Integer, String, Float
from database import Base

class ImageGPSData(Base):
    __tablename__ = "gps_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    distance_km = Column(Float)
    flag = Column(String)