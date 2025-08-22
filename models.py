# models.py (User model only)

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_user_id = Column(String, nullable=False, unique=True, index=True)  # unique + index (avoid duplicate constraint issue)

    created_at = Column(DateTime(timezone=True), server_default=func.now())  # auto timestamp

    # Relationships (cascade delete ensures related records are removed if user is deleted)
    bank_metrics = relationship("BankMetrics", back_populates="user", cascade="all, delete-orphan")
    mobile_money_metrics = relationship("MobileMoneyMetrics", back_populates="user", cascade="all, delete-orphan")
    call_log_metrics = relationship("CallLogMetrics", back_populates="user", cascade="all, delete-orphan")
    gps_data = relationship("GPSData", back_populates="user", cascade="all, delete-orphan")
    image_assets = relationship("ImageAsset", back_populates="user", cascade="all, delete-orphan")
    credit_scores = relationship("CreditScore", back_populates="user", cascade="all, delete-orphan")