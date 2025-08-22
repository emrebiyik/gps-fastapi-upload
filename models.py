from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_user_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    bank_metrics = relationship("BankMetrics", back_populates="user")
    mobile_metrics = relationship("MobileMoneyMetrics", back_populates="user")
    calllog_metrics = relationship("CallLogMetrics", back_populates="user")
    gps_data = relationship("GPSData", back_populates="user")
    assets = relationship("ImageAsset", back_populates="user")
    scores = relationship("CreditScore", back_populates="user")


class BankMetrics(Base):
    __tablename__ = "bank_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    income_avg_3m = Column(Float, nullable=True)
    average_balance = Column(Float, nullable=True)
    net_cash_flow_90d = Column(Float, nullable=True)
    bounced_txn_90d = Column(Integer, nullable=True)
    overdraft_days_90d = Column(Integer, nullable=True)
    statement_period_days = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="bank_metrics")


class MobileMoneyMetrics(Base):
    __tablename__ = "mobile_money_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mm_txn_90d = Column(Integer, nullable=True)
    mm_volume_90d = Column(Float, nullable=True)
    mm_active_days_90d = Column(Integer, nullable=True)
    avg_ticket_90d = Column(Float, nullable=True)
    last_txn_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="mobile_metrics")


class CallLogMetrics(Base):
    __tablename__ = "calllog_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    unique_contacts_30d = Column(Integer, nullable=True)
    call_days_30d = Column(Integer, nullable=True)
    incoming_outgoing_ratio_30d = Column(Float, nullable=True)
    airtime_spend_30d = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="calllog_metrics")


class GPSData(Base):
    __tablename__ = "gps_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    flag = Column(String, nullable=True)   # "normal" / "abnormal"
    taken_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="gps_data")


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_type = Column(String, nullable=False)
    estimated_value = Column(Float, nullable=True)
    image_verified = Column(Integer, default=0)  # boolean stored as int
    source_image = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="assets")


class CreditScore(Base):
    __tablename__ = "credit_scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    score = Column(Integer, nullable=False)
    decision = Column(String, nullable=False)
    explanation_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scores")


__all__ = [
    "User",
    "BankMetrics",
    "MobileMoneyMetrics",
    "CallLogMetrics",
    "GPSData",
    "ImageAsset",
    "CreditScore"
]