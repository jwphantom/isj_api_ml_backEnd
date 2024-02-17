from datetime import datetime
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi_sqlalchemy import DBSessionMiddleware, db

from app.models.models import Image as ModelImage

from app.schema.request import Image as SchemaImage

from app.models.models import Analyse as ModelAnalyse


from app.utils.utils import (
    base64_to_image,
    predictDiseaseBananas,
    recognitionBanane,
    regBase64,
    make_prediction_mri,
    recognitionIRM,
)

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


@router.post("/analyse-plante")
async def analyse_image_base64(image_data: SchemaImage):
    try:
        # verificatiion si on a reçu un lien base64 valide
        valideBase64 = regBase64(image_data.base64)

        if not valideBase64:
            return {"base_64": "Lien Base64 incorrect"}

        # verifcation si c'est une plante qu'on a uploader
        found_plantain = recognitionBanane(image_data.base64)

        if not found_plantain:
            return {"found_plantain": False}

        image = base64_to_image(image_data.base64)

        prediction = predictDiseaseBananas("ml/model.h5", "ml/labels.txt", image)

        date_analyse = datetime.now()

        db_analyse = ModelAnalyse(
            img_base64=image_data.base64,
            malade=prediction["malade"],
            gravite=prediction["gravite"],
            categorie=prediction["categorie"],
            precision=str(prediction["precision"]),
            time_created=date_analyse,
        )

        db.session.add(db_analyse)

        try:
            db.session.commit()
        except Exception as e:
            print(f"Une exception s'est produite : {str(e)}")

        return prediction

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyse-mri")
async def analyse_mri_base64(image_data: SchemaImage):
    try:
        # verificatiion si on a reçu un lien base64 valide
        valideBase64 = regBase64(image_data.base64)

        if not valideBase64:
            return {"base_64": "Lien Base64 incorrect"}

        image = base64_to_image(image_data.base64)

        # verifcation si c'est une plante qu'on a uploader
        found_irm = recognitionIRM(
            "ml/irm/irm_recognition.h5", "ml/irm/labels.txt", image
        )

        if not found_irm:
            return {"found_irm": False}

        analyse = make_prediction_mri(
            "ml/brain-scan.h5",
            image_data.base64,
        )

        analyse["found_irm"] = True

        return analyse

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
