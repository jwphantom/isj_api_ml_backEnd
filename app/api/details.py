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


@router.get("/{item_id}")
async def details_by_id(item_id: str):
    try:
        # Recherchez l'entrée par ID
        details = (
            db.session.query(ModelAnalyse).filter(ModelAnalyse.id == item_id).first()
        )

        if details is None:
            raise HTTPException(status_code=404, detail="L'entrée n'a pas été trouvée")

        return details

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
