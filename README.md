# Metastatic Tissue Classifier

![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-green)
![AWS](https://img.shields.io/badge/AWS-Lambda-orange)
![License](https://img.shields.io/badge/License-CC0_1.0-blue)

The vast majority of cancer-related deaths are attributed to metastatic spread, making detection a critical task in pathology. This project implements a Deep Learning classifier (ResNet-18) trained on the **PCam (PatchCamelyon)** dataset to identify metastatic tissue in histopathologic scans of lymph node sections. 

The project is deployed using a **serverless architecture**, leveraging AWS Lambda for inference and Streamlit for the user interface, ensuring a secure and scalable solution.

---

## Features

- **Deep Learning Inference:** Optimized ONNX model running on AWS Lambda.
- **Serverless Backend:** Fully containerized with Docker and hosted on ECR.
- **Secure Communication:** Uses **Direct Boto3 Invocation** (No public HTTP endpoints) for enhanced security.
- **Interactive UI:** Streamlit-based web interface with sidebar sample testing.
- **Efficient Deployment:** Managed with `uv` for lightning-fast dependency resolution.

---

## Project Structure

```text
.
├── README.md
├── app.py                              # Streamlit Web Application
├── data
│   ├── sample_img                      # Small sample images for testing in the UI
├── models
│   ├── best_model.onnx                 # Optimized model (Baked into Docker image)
├── notebook
│   ├── Dockerfile                      # Container config for AWS Lambda
│   ├── export_onnx.py                  # Script for exporting `.pth` file to ONNX format
│   ├── lambda_function.py              # The Lambda entry point for inference
│   ├── notebook.ipynb                  # EDA, training history and evaluation
│   ├── test_api.py                     # Test local deployment
│   └── train.py                        # Model training and optimization
├── pyproject.toml                      # Project Dependencies
└── uv.lock                             # Dependency lockfile

```

---

## Security Architecture
This project uses a Private Cloud approach:

1. No Public Endpoint: The Lambda function has no public URL or API Gateway.

2. Identity-Based Access: The frontend communicates via the AWS SDK (Boto3).

3. Signed Requests: Every request must be signed with valid IAM Access Keys, preventing unauthorized model usage.

---

## Installation
1. Clone the repository
```bash
git clone <your-repository-url>
cd metastatic-tissue-classifier
```

2. Setup Environment
```bash
uv sync --locked
```

---

## Usage

### Option 1: Live Web App (Recommended)
The easiest way to test the classifier is via the hosted Streamlit interface.
**[URL]**

### Option 2: Local Docker Testing
If you wish to test the model locally without AWS, run the containerized backend using Docker:
1. Pull and Run the container:
```bash
docker build -f ./notebooks/Dockerfile -t metastatic-classifier .
docker run -it --rm -p 9000:8080 metastatic-classifier
```
2. Test the API
Open a second terminal window and run the test script
```bash
uv run ./notebooks/test_api.py
```
*Note: This simulates the Lambda environment locally on port 9000*

---

## Dataset & References
- Dataset: https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon/data
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962
- [2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. doi:jama.2017.14585

---

## License

CC0: Public Domain – see LICENSE file for details.