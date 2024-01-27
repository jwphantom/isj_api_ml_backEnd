import requests
import json
from pprint import pprint

API_KEY = "2b10W9mT7TSqCDze24CizQO"  # Votre clé API
PROJECT = "all"
LANG = "fr"

api_endpoint = (
    f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}&lang={LANG}"
)

image_path_1 = "media/temp_image.jpg"
image_data_1 = open(image_path_1, "rb")

# Si vous n'avez qu'une seule image, spécifiez un seul organe
data = {"organs": ["leaf"]}  # Ici, nous supposons que l'image est celle d'une feuille

files = [("images", (image_path_1, image_data_1))]

response = requests.post(api_endpoint, files=files, data=data)
json_result = response.json()

# Vérifiez si la réponse contient des informations sur la banane plantain
found_plantain = False

bestMatch = json_result.get("bestMatch")

if bestMatch.startswith("Musa"):
    print(bestMatch)
