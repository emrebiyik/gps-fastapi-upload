from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImageGPSData(Base):
    __tablename__ = 'image_gps_data'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    distance_from_previous_km = Column(Float, nullable=True)
    flag = Column(String, nullable=True)