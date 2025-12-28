import streamlit as st
import boto3
import base64
import json
from PIL import Image
import io
import os

# Initialize the AWS client
## it will look for local keys first then Streamlit Secrets as fallback strategy
def get_lambda_client():
    try:
        return boto3.client('lambda', region_name='eu-north-1')
    except Exception:
        # if in the cloud then look for streamlit secrets dashboard
        return boto3.client(
            aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'],
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "eu-north-1")
        )

lambda_client = get_lambda_client()

st.set_page_config(page_title='Tissue Classifer', page_icon = '🔬')
st.title("Metastatic Tissue Classifier")

# ========== 
# Create a sidebar for sample images
# ==========
st.sidebar.header('Sample Images')
st.sidebar.write('Download these to test the classifier:')
sample_dir = "./data/sample_img/"
if os.path.exists(sample_dir):
    for sample in os.listdir(sample_dir):
        if sample.endswith(('.png', '.jpg')):
            with open(os.path.join(sample_dir, sample), 'rb') as f:
                st.sidebar.download_button(
                    label=f'Download {sample}',
                    data = f,
                    file_name = sample,
                    mime = 'image/png'
                )

# ========== 
# Main Interface
# ==========
uploaded_file = st.file_uploader("Upload patch", type = ['png', 'jpg'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Target Patch', width = 250)

    if st.button('Classify'):
        with st.spinner('Invoking Lambda via Secure AWS SDK...'):
            # convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format='PNG')
            img_str = base64.b64encode(buffered.getvalue()).decode()

            payload = {'body': json.dumps({'image': img_str})}

            try:
                # Direct invoke
                response = lambda_client.invoke(
                    FunctionName = 'metastatic-tissue-classifier', # the name of the lambda
                    InvocationType = 'RequestResponse',
                    Payload = json.dumps(payload)
                )

                if response['StatusCode'] == 200:
                    # parse the payload object
                    response_payload = json.loads(response['Payload'].read())
                    # unpack the response body
                    inner_body = json.loads(response_payload['body'])

                    pred = inner_body['prediction']
                    prob = inner_body['probability']

                    st.divider()
                    if pred == 'Metastatic':
                        st.error(f"**Result:** {pred}")
                    else:
                        st.success(f"**Result:** {pred}")

                    st.metric('Probability', f'{prob:.2%}')
                else:
                    st.error(f"AWS Error: {response['StatusCode']}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")