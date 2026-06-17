"""
SQLAlchemy ORM Models
Production-grade database schema with proper indexes.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from config.settings import SETTINGS

Base = declarative_base()


class EarthquakeEvent(Base):
    __tablename__ = "earthquake_events"

    event_id = Column(String(100), primary_key=True)
    event_time = Column(DateTime(timezone=True))
    region_name = Column(String(100), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    magnitude = Column(Float, index=True)
    depth_km = Column(Float)
    severity_label = Column(String(50))
    place = Column(Text)


def create_tables():
    """Create all database tables."""
    engine = create_engine(SETTINGS.database.url)
    Base.metadata.create_all(engine)
