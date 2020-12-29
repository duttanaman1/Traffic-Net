from io import open
import requests
import shutil
from zipfile import ZipFile
from imageai.Prediction.Custom import ModelTraining, CustomImagePrediction
import os

import json
import datetime

import smtplib
import ssl
import pyrebase
import base64
#from tensorflow.python import pywrap_tensorflow

execution_path = os.getcwd()

SOURCE_PATH = "https://github.com/OlafenwaMoses/Traffic-Net/releases/download/1.0/trafficnet_dataset_v1.zip"
FILE_DIR = os.path.join(execution_path, "trafficnet_dataset_v1.zip")
DATASET_DIR = os.path.join(execution_path, "trafficnet_dataset_v1.zip")


def download_traffic_net():
    if (os.path.exists(FILE_DIR) == False):
        print("Downloading trafficnet_dataset_v1.zip")
        data = requests.get(SOURCE_PATH,
                            stream=True)

        with open(FILE_DIR, "wb") as file:
            shutil.copyfileobj(data.raw, file)
        del data

        extract = ZipFile(FILE_DIR)
        extract.extractall(execution_path)
        extract.close()


def train_traffic_net():
    download_traffic_net()

    trainer = ModelTraining()
    trainer.setModelTypeAsResNet()
    trainer.setDataDirectory("trafficnet_dataset_v1")
    trainer.trainModel(num_objects=4, num_experiments=200,
                       batch_size=32, save_full_model=True, enhance_data=True)


def write(result):
    with open("data.json", "r+") as file:
        data = json.load(file)

        data.append(result)
        file.seek(0)
        json.dump(data, file)


def run_predict():
    predictor = CustomImagePrediction()
    predictor.setModelPath(
        model_path="trafficnet_resnet_model_ex-055_acc-0.913750.h5")
    predictor.setJsonPath(model_json="model_class.json")
    predictor.loadFullModel(num_objects=4)

    predictions, probabilities = predictor.predictImage(
        image_input="images/traff.jpg", result_count=4)
    for prediction, probability in zip(predictions, probabilities):
        print(prediction, " : ", probability)
        result["accident"][prediction] = probability
        # otus thresholding of 80
        if probability > 80:
            result["accident_result"][prediction] = True
        else:
            result["accident_result"][prediction] = False
    if result["accident_result"]["Accident"] == True or result["accident_result"]["Fire"] == True:
        sendemail()
    print(result)
    write(result)


# Un-comment the line below to train your model
# train_traffic_net()

result = {}


def jsonwrite():
    currdate = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    result["date"] = currdate
    result["accident"] = {}
    result["accident_result"] = {}


def sendemail():
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "tarpproject79@gmail.com"
    receiver_email = "duttanaman1@gmail.com"
    password = 'Sunny9849'
    message = "Accident has been detected"

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def serverimagedownload():
    config = {
        "apiKey": "AIzaSyBW_vjnHLlNenVON-4Gko60n78_0BIukjs",
        "authDomain": "accident-detection-c5938.firebaseapp.com",
        "databaseURL": "https://accident-detection-c5938.firebaseio.com",
        "projectId": "accident-detection-c5938",
        "storageBucket": "accident-detection-c5938.appspot.com",
        "messagingSenderId": "927567326901",
        "appId": "1:927567326901:web:29c0a73d04a32b6bd258fc",
        "measurementId": "G-MBX30N91P8"
    }

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    img_url = db.child('Image').get().val()
    img_data = b' img_url.split(",")[-1]'
    with open("images/online.jpeg", "wb") as fh:
        fh.write(base64.decodebytes(img_data))


jsonwrite()
# serverimagedownload()

# Un-comment the line below to run predictions
run_predict()
