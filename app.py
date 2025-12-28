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

st.set_page_config(page_title='Tissue Classifer', page_icon = '🔬', layout='wide')

# ========== 
# Create a sidebar for sample images
# ==========
st.sidebar.header('Test Images')
st.sidebar.write('Click on a test image to load it directly:')
sample_dir = "./data/sample_img/"
selected_sample = None

if os.path.exists(sample_dir):
    # sort files to keep 'non-metastatic' and 'metastatic' in consistent order
    samples = sorted([f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))])

    for sample in samples:
        path = os.path.join(sample_dir, sample)
        img_display = Image.open(path)

        # column layout
        col1, col2 = st.sidebar.columns([1,2])
        with col1:
            st.image(img_display, use_container_width=True)
        with col2:
            # add short labels
            label = 'Negative' if 'non' in sample else 'Positive'
            if st.button(f"Load {label}", key = sample):
                selected_sample = img_display
        st.sidebar.divider()

# ==========
# Data Source Link
# ==========
st.sidebar.subheader("Data Source")
st.sidebar.caption(
    """
    Test patches are sources from the **Metastatic Tissue Classification - PatchCamelyon** Dataset on Kaggle.
    [Link to dataset](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon)
    """
)

# ========== 
# Main Interface
# ==========
st.title("Metastatic Tissue Classifier")

# Project and Data expander
with st.expander("About this Project and Dataset"):
    st.markdown(
        """
        ### Project Overview
        Deep Learning classifier using ResNet-18 to detect metastatic tissue in histopathologic scans from PatchCamelyon. 
        Features a Streamlit UI and a private, serverless AWS Lambda backend. 
        Capstone project for the 2025 cohort of the Machine Learning Zoomcamp course.

        ### Dataset Details
        The images used are part of the **Metastatic Tissue Classification - PatchCamelyon** Dataset on Kaggle.
        It contains 327.680 color images (96 x 96 pixels) part of the PatchCamelyon benchmark (PCAM). 
        The images represent histopathologic scans of lymph node sections, with annotation to indicate the presence of metastatic tissue.

        * **Data Citation:**
            * https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon
            * [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962
            * [2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. doi:jama.2017.14585
        """
    )

st.info(
    """
    ** How to use:** 
    1. **Upload** a histopathologic slide patch (96x96 pixels) below.
    2. Alternatively, **select a sample** from the sidebar on the left.
    3. Click **'Classify'** to run the ResNet-18 model via AWS Lambda.
    """
)
uploaded_file = st.file_uploader("Upload patch", type = ['png', 'jpg'])

# Determine which image to use
input_image = None
if uploaded_file:
    input_image = Image.open(uploaded_file)
elif selected_sample:
    input_image = selected_sample

# Display selected image and classify
if input_image:
    st.image(input_image, caption = 'Target Patch', width = 250)

    if st.button('Classify Slide', type = 'primary'):
        with st.spinner('Invoking Lambda via Secure AWS SDK...'):
            # convert image to base64
            buffered = io.BytesIO()
            input_image.save(buffered, format='PNG')
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
                    st.progress(prob)
                else:
                    st.error(f"AWS Error: {response['StatusCode']}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")