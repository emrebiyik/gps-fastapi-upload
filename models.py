# models.py
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship
from database import Base  # Base is defined in database.py

# -----------------------------
# Users
# -----------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_user_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # relationships
    gps_points = relationship("GPSData", back_populates="user", cascade="all, delete-orphan")
    image_assets = relationship("ImageAsset", back_populates="user", cascade="all, delete-orphan")
    bank_metrics = relationship("BankMetrics", back_populates="user", cascade="all, delete-orphan")
    mobile_money_metrics = relationship("MobileMoneyMetrics", back_populates="user", cascade="all, delete-orphan")
    calllog_metrics = relationship("CallLogMetrics", back_populates="user", cascade="all, delete-orphan")
    credit_scores = relationship("CreditScore", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_external_user_id", "external_user_id"),
    )

# -----------------------------
# GPSData (no currency)
# -----------------------------
GPSFlagEnum = Enum("normal", "abnormal", name="gps_flag_enum")

class GPSData(Base):
    __tablename__ = "gps_data"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    filename = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    flag = Column(GPSFlagEnum, nullable=True)
    taken_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="gps_points")

    __table_args__ = (
        Index("ix_gps_user_created", "user_id", "created_at"),
    )

# -----------------------------
# ImageAssets (USD + currency_code)
# -----------------------------
class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    asset_type = Column(String, nullable=False)                # e.g., "motorbike", "fridge"
    estimated_value = Column(Float, nullable=True)             # USD
    currency_code = Column(String, default="USD", nullable=False)  # ISO-4217 (e.g., "USD")
    image_verified = Column(Boolean, default=False, nullable=False)
    source_image = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="image_assets")

    __table_args__ = (
        Index("ix_assets_user_created", "user_id", "created_at"),
    )

# -----------------------------
# BankMetrics (USD + currency_code)
# -----------------------------
class BankMetrics(Base):
    __tablename__ = "bank_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    income_avg_3m = Column(Float, nullable=True)               # USD
    average_balance = Column(Float, nullable=True)             # USD
    net_cash_flow_90d = Column(Float, nullable=True)           # USD
    bounced_txn_90d = Column(Integer, nullable=True)
    overdraft_days_90d = Column(Integer, nullable=True)
    statement_period_days = Column(Integer, nullable=True)
    currency_code = Column(String, default="USD", nullable=False)   # ISO-4217

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="bank_metrics")

    __table_args__ = (
        Index("ix_bank_user_created", "user_id", "created_at"),
    )

# -----------------------------
# MobileMoneyMetrics (USD + currency_code)
# -----------------------------
class MobileMoneyMetrics(Base):
    __tablename__ = "mobile_money_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    mm_txn_90d = Column(Integer, nullable=True)
    mm_volume_90d = Column(Float, nullable=True)               # USD
    mm_active_days_90d = Column(Integer, nullable=True)
    avg_ticket_90d = Column(Float, nullable=True)              # USD
    last_txn_at = Column(DateTime, nullable=True)
    currency_code = Column(String, default="USD", nullable=False)    # ISO-4217

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="mobile_money_metrics")

    __table_args__ = (
        Index("ix_mobile_user_created", "user_id", "created_at"),
    )

# -----------------------------
# CallLogMetrics (USD + currency_code where applicable)
# -----------------------------
class CallLogMetrics(Base):
    __tablename__ = "calllog_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    unique_contacts_30d = Column(Integer, nullable=True)
    call_days_30d = Column(Integer, nullable=True)
    incoming_outgoing_ratio_30d = Column(Float, nullable=True)
    airtime_spend_30d = Column(Float, nullable=True)           # USD
    currency_code = Column(String, default="USD", nullable=False)    # ISO-4217

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="calllog_metrics")

    __table_args__ = (
        Index("ix_call_user_created", "user_id", "created_at"),
    )

# -----------------------------
# CreditScores (no currency)
# -----------------------------
DecisionEnum = Enum(
    "approve_500", "approve_400", "approve_150", "deny", "review",
    name="decision_enum"
)

class CreditScore(Base):
    __tablename__ = "credit_scores"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    score = Column(Integer, nullable=False)
    decision = Column(DecisionEnum, nullable=False)
    explanation_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="credit_scores")

    __table_args__ = (
        Index("ix_score_user_created", "user_id", "created_at"),
    )