import uvicorn

from fastapi import FastAPI, UploadFile, File,Form
from fastapi.middleware.cors import CORSMiddleware
import fastai
from fastai.vision.all import *
print(fastai.__version__)

from PIL import Image
import os
import pathlib
from pathlib import Path

if platform.system() == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
else: pathlib.PosixPath = pathlib.WindowsPath

import requests
from time import time

app = FastAPI()
origins = [
    "https://supachai1998.github.io/test_project_64",
    "https://supachai1998.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "https://localhost",
    "https://localhost:3000",
    "https://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def target_downloand(url,name):
    if not os.path.isfile(name):
        destination = name
        print(":: --- try download full_model --- ::")
        download_file_from_google_drive(url, destination)
        print(":: ---  downloaded --- ::")

    os.chmod(Path(".") , 0o777)


target_downloand("1uW_3446XPrKJ65Oc6EBWFvSxFpPgY6o8","full_model.pkl")
target_downloand("10TKLUOmgXNWRCuiBt-N5BlRnJkewGx_E","soft_model.pkl")
target_downloand("1iz1TPVqW6JtCHjg1gwnzByfHrXUs_ivY","isFood.pkl")

food_model = load_learner(Path("isFood.pkl"))
full_model = load_learner(Path("full_model.pkl"))
soft_model = load_learner(Path("soft_model.pkl"))
def check_food(data) -> Image.Image:
    classname, index, confident = food_model.predict(data)
    confident *= 100
    return classname, index, confident

def full_predict(data) -> Image.Image:
    classname, index, confident = full_model.predict(data)
    confident *= 100
    return classname, index, confident

@app.get('/')
def index():
    return {'Server': 'Image Classification API'}

@app.post("/backend/predict")
async def predict(file: UploadFile   = File(...)):
    filename = file.filename
    _ , ext = os.path.splitext(filename)
    del _
    if ext.lower() not in [".jpg", ".png", ".jpeg"]: return {
            "type" : "ไม่ใช่ภาพ",
        }
    del ext
    temp_file = await file.read()
    classname, index, confident = check_food(temp_file)
    _confident = confident[index]
    _confident = float(_confident)
    print("check food", classname, _confident)
    if classname.split(".")[-1].lower() == "nonfood" or (classname.split(".")[-1].lower() == "food" and _confident < 70) : return {
        "type" : "ไม่ใช่อาหาร",
    }
    classname, index, confident = full_predict(temp_file)
    _confident = confident[index]
    _confident = float(_confident)
    name = classname.split("-")[-1]
    print( classname, _confident)
    del file,temp_file
    return {
        "type" : "อาหาร",
        "predict_topic": name,
        "confident_percent" : _confident,      
    }

@app.get("/backend/labels")
async def labels():
    return full_model.dls.vocab

@app.post("/backend/line_push")
async def send_msg_line(uid: str = Form(...), msg: str = Form(...),):
    url = "https://api.line.me/v2/bot/message/push"

    payload = json.dumps({
    "to": uid,
    "messages": [
        {
        "type": "text",
        "text": msg
        }
    ]
    })
    headers = {
        'Authorization': 'Bearer jmEgNZJqxsSGrwSdTBzWEu/ywMimnzBFGZ7DWscNY3vOlfPYQKZlEHYd87RLG9lnaT1wSKoIhv7vuJ7ks/vtXN4O5mQh+zsoNia4rOICz5honFl0XrGkw6nLLDckflUd9fdBkJlbeesvXgEN8712kQdB04t89/1O/w1cDnyilFU=',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.status_code

if __name__ == '__main__':
    uvicorn.run(app, host='192.168.43.65', port=8000)
# conda activate fastai 
#  & D:/miniconda/envs/fastai/python.exe  
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload --debug
# heroku maintenance:off


# git add .
# git commit -m "edit predict_topic"
# git push heroku master 

# heroku repo:gc --app ml-test-f-api-2022
# heroku repo:purge_cache --app ml-test-f-api-2022
# git gc --aggressive