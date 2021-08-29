import cv2
import base64
import numpy as np
from io import BytesIO
import tensorflow as tf
from .forms import ImageForm
from django.shortcuts import render
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/model_disease2.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile=False)

def predict(img_path, model, img_size = (256,256)):
    input_image = tf.io.read_file(img_path)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.resize(input_image, img_size) 
    input_image = tf.cast(input_image, tf.float32) / 255.0
    pred_mask = (model.predict(input_image[tf.newaxis, ...])[0,:,:,0] > 0.5).astype(np.uint8)
    pred_mask = pred_mask.reshape(img_size[0], img_size[1], 1)

    mask_on_img = cv2.add(input_image.numpy(),  np.zeros(np.shape(input_image), dtype= np.uint8),
                          mask = pred_mask, dtype=cv2.CV_64F)
    return input_image, pred_mask, mask_on_img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/png;base64,'+data64.decode('utf-8')

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.instance
            form.save()

            # Making Prediction on image
            input_img, pred_mask, mask_on_img = predict(
                "media/"+str(img_obj.image), model)
            
            input_image = tf.keras.preprocessing.image.array_to_img(
                input_img)
            input_image_uri = to_data_uri(input_image)

            mask_image = tf.keras.preprocessing.image.array_to_img(pred_mask)
            mask_uri = to_data_uri(mask_image)

            mask_on_image = tf.keras.preprocessing.image.array_to_img(
                mask_on_img)
            mask_on_img_uri = to_data_uri(mask_on_image)
            
            context = {"form": form, 'img_obj': img_obj, 'input_image_uri': input_image_uri,
                       'mask_uri': mask_uri, 'mask_on_img_uri': mask_on_img_uri}

            return render(request, 'Home/index.html', context = context)
            
    else:
        form = ImageForm()
    
    return render(request, 'Home/index.html', {"form": form})