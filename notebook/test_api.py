import base64
import requests
import json

def test_container():
    url = "http://localhost:9000/2015-03-31/functions/function/invocations"
    image_path = './data/sample_img/sample_0_non-metastatic.png'

    with open(image_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    payload = {
        'body': json.dumps({
            'image': img_b64
        })
    }

    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")

if __name__ == '__main__':
    test_container()