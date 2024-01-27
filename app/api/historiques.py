from datetime import datetime
from sqlalchemy import func
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi_sqlalchemy import DBSessionMiddleware, db

from app.models.models import Image as ModelImage

from app.schema.request import Image as SchemaImage

from app.models.models import Analyse as ModelAnalyse


from app.utils.utils import base64_to_image

from keras.models import load_model

from PIL import Image, ImageOps

import numpy as np

import base64
from io import BytesIO


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


import os
from dotenv import load_dotenv

router = APIRouter()


@router.get("/all")
async def analyse_image_base64():
    try:
        historiques = db.session.query(ModelAnalyse).all()
        return historiques

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/by_date/{date}")
async def analyse_by_date(date: str):
    try:
        # Convertissez la date en un objet datetime
        date_datetime = datetime.strptime(date, "%Y-%m-%d")

        # Recherchez les entrées avec la date spécifiée
        historiques = (
            db.session.query(ModelAnalyse)
            .filter(func.date(ModelAnalyse.time_created) == date_datetime.date())
            .all()
        )

        return historiques

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
