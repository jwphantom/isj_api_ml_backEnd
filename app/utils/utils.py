import base64
from io import BytesIO
from PIL import Image, ImageOps

from keras.models import load_model

import numpy as np

import os

import requests
import json

import re

from keras.models import Model

from keras.utils import to_categorical

import keras

import tensorflow as tf

import matplotlib as mpl


def base64_to_image(base64_string):
    # Splitting the string to remove the MIME type prefix
    if "," in base64_string:
        header, base64_string = base64_string.split(",", 1)

    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # image.save(save_path)  # Save the image
        return image
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image : {e}")
        raise e


def predictDiseaseBananas(model, label, image):
    # Load the model
    model = load_model(model, compile=False)

    class_names = open(label, "r").readlines()  # Resizing and cropping the image

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    gravite_dict = {
        "Cordana": "Modérée à sévère",
        "Healthy": "Aucune",
        "Pestalotiopsis": "Modérée",
        "Sigatoka": "Très élevée",
    }

    gravite = gravite_dict.get(class_name[2:].strip(), "Inconnue")

    malade = True

    if int(class_name[:1]) == 1:
        malade = False

    return {
        "malade": malade,
        "categorie": class_name[2:].strip(),
        "precision": float(confidence_score),
        "gravite": gravite,
    }


def recognitionBanane(base64_img):
    API_KEY = "2b10W9mT7TSqCDze24CizQO"  # Votre clé API
    PROJECT = "all"
    LANG = "fr"
    IMG_PATH = "media/temp_image.jpg"

    image = base64_to_image(base64_img)
    image.save(IMG_PATH, format="JPEG")

    api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}&lang={LANG}"

    image_data = open(IMG_PATH, "rb")

    data = {"organs": ["leaf"]}

    files = [("images", (IMG_PATH, image_data))]

    response = requests.post(api_endpoint, files=files, data=data)
    json_result = response.json()

    # Vérifiez si la réponse contient des informations sur la banane plantain
    found_plantain = False

    if json_result.get("status_code") == 404:
        return False

    for result in json_result["results"][
        :2
    ]:  # Examiner seulement les deux premiers résultats
        scientific_name = result["species"]["scientificName"]
        if scientific_name.startswith("Musa") or scientific_name.startswith("Canna"):
            return True
    return False


def regBase64(base64_url):
    # Regex pour détecter une chaîne Base64 d'image
    regex_base64_url = r"data:image\/(png|jpg|jpeg|gif);base64,[A-Za-z0-9+/=]+"

    # Test de la chaîne avec la regex
    if re.match(regex_base64_url, base64_url):
        return True

    return False


last_conv_layer_name = "Top_Conv_Layer"


# chargement d'une image et la convertir en un tableau NumPy adapté pour être traité par le modèle
def get_img_array(img, size=(224, 224)):
    img = img.resize(size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


"""
Fonction qui génère une "carte thermique" (heatmap) en utilisant la technique
Grad-CAM, permettant de visualiser quelles régions de l'image sont importantes
pour la prédiction du modèle.
"""


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name=last_conv_layer_name, pred_index=None
):
    # Tout d'abord, nous créons un modèle qui associe l'image d'entrée aux activations de la dernière couche de conv ainsi que les prédictions de sortie
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Ensuite, nous calculons le gradient de la classe prédite la plus élevée pour notre image d'entrée par rapport aux activations de la dernière couche de conv.
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Il s'agit du gradient du neurone de sortie (prédit ou choisi en haut) par rapport à la carte des caractéristiques de sortie de la dernière couche de conv.
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Il s'agit d'un vecteur dont chaque entrée représente l'intensité moyenne du gradient sur un canal spécifique de la carte des caractéristiques.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    """
    Nous multiplions chaque canal du tableau de la carte des caractéristiques
    par "l'importance de ce canal" par rapport à la classe prédite la plus élevée,
    puis nous additionnons tous les canaux pour obtenir l'activation de la classe
    de la carte thermique.
    """
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # À des fins de visualisation, nous normaliserons également la carte thermique entre 0 et 1.
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


"""
Fonction qui sauvegarde et affiche éventuellement l'image originale superposée
avec la heatmap générée par Grad-CAM.
"""


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Convertir l'objet image en tableau NumPy
    img = keras.preprocessing.image.img_to_array(img)

    # Rééchelonner la carte thermique sur une plage de 0 à 255
    heatmap = np.uint8(255 * heatmap)

    # Utiliser le jet colormap pour colorer la carte thermique
    jet = mpl.cm.get_cmap("jet")

    # Utiliser les valeurs RVB de la carte des couleurs
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Créer une image avec une carte thermique colorée RVB
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superposer la carte thermique à l'image originale
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Convertir l'image superposée en chaîne base64
    buffered = BytesIO()
    superimposed_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_base64}"


# Fonction qui écode les prédictions du modèle en noms de classes lisibles.
def decode_predictions(preds):
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    prediction = preds
    index = np.argmax(prediction)
    class_name = classes[index]
    confidence_score = prediction[0][index]

    if confidence_score == 1.0:
        confidence_score = confidence_score - 0.001

    return (class_name, (confidence_score * 100))


"""
Fonction qui encapsule tout le processus : elle prend une image, effectue une
prédiction avec le modèle, crée une heatmap avec Grad-CAM, et sauvegarde/affiche le résultat.
"""


def make_prediction_mri(
    model,
    base64_str,
    last_conv_layer_name=last_conv_layer_name,
):
    model = load_model(model, compile=False)

    # Convertir base64 en objet image
    img = base64_to_image(base64_str)

    # Obtenir le tableau d'image adapté
    img_array = get_img_array(img, size=(224, 224))

    # Effectuer la prédiction
    preds = model.predict(img_array)

    classname, precision = decode_predictions(preds)

    if not (classname == "No Tumor"):
        # Créer la heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Sauvegarder et afficher (si nécessaire) l'image résultante
        img_base64 = save_and_display_gradcam(img, heatmap)

    else:
        img_base64 = base64_str

    return {
        "campath": img_base64,
        "classname": classname,
        "precision": float(precision),
    }


def recognitionIRM(model, label, image):
    # Load the model
    model = load_model(model, compile=False)

    class_names = open(label, "r").readlines()  # Resizing and cropping the image

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    if int(class_name[:1]) == 0:
        return True

    return False
