import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import load_model


from io import BytesIO
from PIL import Image
import requests


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


# Model saved with Keras model.save()
model = load_model('models/model_saved.h5',compile= True)
model2 = load_model('models/model_saved_color.h5',compile=True)
print('Model loaded.')

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def loadImage(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    return img


def model_predict(path,model):
    #path = input("Enter File Path: (alternatively type Quit to close) ")
    if path=='Quit':
        return 0
    else:
        try:
            try:
                image = load_img(path, target_size=(224, 224))
                img = np.array(image)
            except:
                image= loadImage(path)
                image = image.convert('RGB')
                image = image.resize((224,224), Image.NEAREST)
                img = np.array(image)
            img = img / 255.0
            img = img.reshape(1,224,224,3)
            label = model.predict(img) ### Model prediction
            classes_x=np.argmax(label,axis=1)
            if classes_x==0:
                return ('concrete')
            elif classes_x==1:
                return ('epoxy')
            elif classes_x==2:
                return ('tiles')
            else:
                return ('trimax')

        except Exception as e:
            print (e)
            print ("Something went wrong, please try again")

@app.route("/")
def index():
    # Main page
    return jsonify(result='Go to /send-image/')


@app.route("/send-image/<path:url>")
def image_check(url):
    try:
        try:
            image = load_img(url, target_size=(224, 224))
            img = np.array(image)
        except:
            image= loadImage(url)
            image = image.convert('RGB')
            image = image.resize((224,224), Image.NEAREST)
            img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,224,224,3)
        label = model.predict(img) ### Model prediction
        classes_x=np.argmax(label,axis=1)
        if classes_x==0:
            result =  ('concrete')
        elif classes_x==1:
            result =  ('epoxy')
        elif classes_x==2:
            result = ('tiles')
        else:
            result = ('trimix')

        label = model2.predict(img) ### Model prediction
        classes_x=np.argmax(label,axis=1)
        if classes_x==0:
            color =  ('Dark Floor')
        elif classes_x==1:
            color = ('Light FLoor')
        elif classes_x==2:
            color = ('Medium Floor')
        else:
            color = ('None')


    except Exception as e:
        print (e)
        result = ("Something went wrong, please try again")
        #predict()

    # When you type http://127.1.0.0:5000/send-image/https://sample-website.com/sample-cdn/photo1.jpg to the browser
    # you will se the whole "https://sample-website.com/sample-cdn/photo1.jpg"
    if result =='trimix':
        return jsonify(result=result,color=color)
    else:
        return jsonify(result=result)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)


        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    #app.run(host = '0.0.0.0',port=8080)
    app.run(host = '15.207.197.30',port = 8080)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()