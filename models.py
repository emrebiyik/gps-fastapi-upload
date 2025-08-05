from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImageGPSData(Base):
    __tablename__ = "gps_images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)