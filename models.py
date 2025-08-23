# models.py
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from database import Base


# ---------------------------
# Users
# ---------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # NOT: unique=True zaten benzersiz indeks oluşturur; index=True gereksiz ve çakışma yaratıyordu.
    external_user_id = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional relationships
    bank_metrics = relationship("BankMetrics", back_populates="user", cascade="all, delete-orphan")
    mobile_money_metrics = relationship("MobileMoneyMetrics", back_populates="user", cascade="all, delete-orphan")
    call_log_metrics = relationship("CallLogMetrics", back_populates="user", cascade="all, delete-orphan")
    image_assets = relationship("ImageAsset", back_populates="user", cascade="all, delete-orphan")
    gps_points = relationship("GPSData", back_populates="user", cascade="all, delete-orphan")
    credit_scores = relationship("CreditScore", back_populates="user", cascade="all, delete-orphan")


# ---------------------------
# Bank metrics
# ---------------------------
class BankMetrics(Base):
    __tablename__ = "bank_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    income_avg_3m = Column(Float, nullable=True)
    average_balance = Column(Float, nullable=True)
    net_cash_flow_90d = Column(Float, nullable=True)
    bounced_txn_90d = Column(Integer, nullable=True)
    overdraft_days_90d = Column(Integer, nullable=True)
    statement_period_days = Column(Integer, nullable=True)

    currency_code = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="bank_metrics")


# ---------------------------
# Mobile money metrics
# ---------------------------
class MobileMoneyMetrics(Base):
    __tablename__ = "mobile_money_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    mm_txn_90d = Column(Integer, nullable=True)
    mm_volume_90d = Column(Float, nullable=True)
    mm_active_days_90d = Column(Integer, nullable=True)
    avg_ticket_90d = Column(Float, nullable=True)
    last_txn_at = Column(DateTime, nullable=True)

    currency_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="mobile_money_metrics")


# ---------------------------
# Call log metrics
# ---------------------------
class CallLogMetrics(Base):
    __tablename__ = "call_log_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    unique_contacts_30d = Column(Integer, nullable=True)
    call_days_30d = Column(Integer, nullable=True)
    incoming_outgoing_ratio_30d = Column(Float, nullable=True)
    airtime_spend_30d = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="call_log_metrics")


# ---------------------------
# Image assets
# ---------------------------
class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    asset_type = Column(String, nullable=True)
    estimated_value = Column(Float, nullable=True)
    image_verified = Column(Boolean, nullable=False, default=False)
    source_image = Column(String, nullable=True)

    currency_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="image_assets")


# ---------------------------
# GPS data
# ---------------------------
class GPSData(Base):
    __tablename__ = "gps_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    filename = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    flag = Column(String, nullable=True)  # "normal" / "abnormal"
    taken_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="gps_points")


# ---------------------------
# Credit scores
# ---------------------------
class CreditScore(Base):
    __tablename__ = "credit_scores"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    score = Column(Integer, nullable=False)
    decision = Column(String, nullable=False)  # e.g., "deny", "approve_150", ...
    explanation_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="credit_scores")


__all__ = [
    "User",
    "BankMetrics",
    "MobileMoneyMetrics",
    "CallLogMetrics",
    "ImageAsset",
    "GPSData",
    "CreditScore",
]