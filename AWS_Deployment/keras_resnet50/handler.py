print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzipped')


import json
import keras_applications
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import boto3, os, tempfile 
print('imports end')


keras_applications.imagenet_utils.CLASS_INDEX = json.load(open('imagenet_class_index.json'))

# MODEL_BUCKET_NAME = 'ml-model-1111'
# MODEL_KEY_NAME = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
# IMG_BUCKET_NAME = 'image-upload-111'
# IMG_KEY_NAME = 'tiger.jpg' 

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
model = ResNet50(weights=model_path)
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
        img = image.load_img(tmp_file.name, target_size=(224, 224))
        x   = image.img_to_array(img)
        x   = np.expand_dims(x, axis=0)
        x   = preprocess_input(x)
        tmp_file.close()

        #predict image class and decode prediction
        preds = model.predict(x)
        decode_pred = decode_predictions(preds, top=3)[0]
      
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

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
