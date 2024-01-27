# build a schema using pydantic
from pydantic import BaseModel


class Image(BaseModel):
    uid: str
    base64: str


# class Analyse(BaseModel):
#     uid: str
#     base64: str
#     malade: bool
#     precision: float

#     class Config:
#         orm_mode = True
