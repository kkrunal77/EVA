print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzipped')

import json
import keras_applications
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
import numpy as np
import boto3, os, tempfile 
print('imports end')

#imagenet classes
keras_applications.imagenet_utils.CLASS_INDEX = json.load(open('imagenet_class_index.json'))


MODEL_BUCKET_NAME = os.environ['MODEL_BUCKET_NAME']
MODEL_KEY_NAME = os.environ['MODEL_KEY_NAME']
IMG_BUCKET_NAME = os.environ['IMG_BUCKET_NAME']
print(MODEL_BUCKET_NAME)
print(MODEL_KEY_NAME)
print(IMG_BUCKET_NAME)
# IMG_KEY_NAME = os.environ['IMG_KEY_NAME']
temp_dir = '/tmp'

print('downloading model from AWS...\n')
s3 = boto3.resource('s3')
model_path = os.path.join(temp_dir, MODEL_KEY_NAME)
s3.Bucket(MODEL_BUCKET_NAME).download_file(MODEL_KEY_NAME, model_path)

print('loading model...\n')
model = InceptionV3(weights=model_path)
print("model loaded\n")



def classify(event, context):
    body = {}

    params = event['queryStringParameters']
    if params is not None and 'imageKey' in params:
        image_key = params['imageKey']
        
        #download image
        print("Downloading image...\n")
        tmp_file = tempfile.NamedTemporaryFile(dir=temp_dir)
        img_object = s3.Bucket(IMG_BUCKET_NAME).Object(image_key)
        img_object.download_fileobj(tmp_file)
        print("Image downloaded to", tmp_file.name)

        #load and preprocess image
        inputShape = (299, 299)
        image = load_img(tmp_file.name, target_size=inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        tmp_file.close()

        #predict image class and decode prediction
        preds = model.predict(image)
        decode_pred = imagenet_utils.decode_predictions(preds, top=3)[0]
      
        body['message'] = "OK"
        body['predictions'] = [{"label": pred[1].replace('_',' ').title(), "probability":float(pred[2])} for pred in decode_pred]

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": 'application/json',
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response