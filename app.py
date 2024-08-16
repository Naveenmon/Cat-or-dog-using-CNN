import cv2
from flask import Flask, render_template, request, send_from_directory
import pickle
from jinja2 import escape
import numpy as np
import pandas as pd
import os
import tensorflow as tb
from tensorflow.keras.preprocessing.image import ImageDataGenerator

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    global COUNT
    img = request.files['image']
    img.save('pred/{}.jpg')
    img_arr = cv2.imread('pred/{}.jpg')
    img_arr = cv2.resize(img_arr, (128, 128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128, 128, 3)
    final = [np.array(img_arr)]
    prediction = model.predict(final)
    x = round(prediction[0, 0], 2)
    y = round(prediction[0, 1], 2)
    preds = np.array([x, y])
    COUNT += 1
    return render_template('predict.html',data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', '{}.jpg'.format(COUNT - 1))


if __name__ == '__main__':
    app.run(debug=True, port=5003)
