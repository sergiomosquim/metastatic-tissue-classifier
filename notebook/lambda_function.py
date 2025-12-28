import json
import base64
import io
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

# initialize sessions outside the handler
# it keeps the model in memory across multiple API requests
model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', '.'), 'best_model.onnx')
session = ort.InferenceSession(model_path)

def preprocess(image_bytes):
    # load the image and convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # resize to the expected 96x96 by the model
    img = img.resize((96,96))

    # normalize (ImageNet standard used in training)
    img_arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
    img_arr = (img_arr - mean)/std

    # reorder dimensions from (H, W, C) to (C, H, W)
    img_arr = img_arr.transpose(2,0,1)
    
    # add batch dimension (1, 3, 96, 96) format
    return np.expand_dims(img_arr.astype(np.float32), axis = 0)

def lambda_handler(event, context):
    try:
        # parse body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        image_64 = body.get('image')
        if not image_64:
            return {
                'statusCode': 400, 
                'body': json.dumps({
                    'error': 'No image key in body'
                })
            }
        
        # clean base64 string (handles potential prefixes)
        if "," in image_64:
            image_64 = image_64.split(",")[1]
        
        # decode and predict
        image_bytes = base64.b64decode(image_64)
        input_data = preprocess(image_bytes)

        outputs = session.run(None, {'input': input_data})
        logits = outputs[0]
        prob = 1/(1+np.exp(-logits))
        prob = float(prob[0][0])

        return {
            'statusCode': 200, 
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'prediction': 'Metastatic' if prob > 0.5 else 'Non-Metastatic',
                'probability': round(prob,4)
            })
        }
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }