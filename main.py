from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware
from app.api import analyse, historiques, details


from fastapi.middleware.cors import CORSMiddleware


import os
from dotenv import load_dotenv

load_dotenv()

origins = [
    "http://localhost:5173",
    "https://agri-analysishub.netlify.app",
    "http://localhost:3000",
    "https://brain-scan-ai.vercel.app",
]


app = FastAPI()

app.include_router(analyse.router, prefix="/api/analyse", tags=["analyse"])

app.include_router(historiques.router, prefix="/api/historique", tags=["historique"])

app.include_router(details.router, prefix="/api/details", tags=["details"])


# to avoid csrftokenError
# to avoid csrftokenError
app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
