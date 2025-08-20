import requests

url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
out = "emotion_model.h5"

print("Téléchargement du modèle...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
print("Modèle téléchargé:", out)pip install requests
