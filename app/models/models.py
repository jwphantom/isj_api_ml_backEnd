import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Image(Base):
    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    img_base64 = Column(String)


class Analyse(Base):
    __tablename__ = "analyses"
    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True
    )
    img_base64 = Column(String)
    malade = Column(Boolean)
    precision = Column(String)
    categorie = Column(String)
    gravite = Column(String)
    time_created = Column(DateTime(timezone=True))
