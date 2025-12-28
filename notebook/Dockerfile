# =============
# BUILDER STAGE
# =============
FROM public.ecr.aws/lambda/python:3.11 AS builder

# install uv (amazon linux uses yum instead of apt-get)
RUN yum install -y gzip tar
## Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
## Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
## Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /builder

# Copy dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies
## Export to temporary file, also ignoring 'train' group
RUN uv export --format requirements-txt --no-dev --no-editable > requirements.txt
## Install using requirements.txt file
RUN uv pip install --no-cache --target /install -r requirements.txt

# =============
# FINAL STAGE
# =============
FROM public.ecr.aws/lambda/python:3.11

## Open Container Initiative (OCI) Standard Metadata
LABEL org.opencontainers.image.title="PCam Metastatic Tissue Classifier" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.description="ResNet-18 based model for detecting metastatic cancer in histopathology patches" \
    org.opencontainers.image.authors="Sergio Mosquim Junior <sergiomosquim@live.com>" \
    org.opencontainers.image.source="https://github.com/sergiomosquim/metastatic_tissue_classifier" \
    org.opencontainers.image.licenses="CC0-1.0"

# Copy installed dependencies
COPY --from=builder /install ${LAMBDA_TASK_ROOT}

# Copy model and necessary code
COPY ./models/best_model.onnx ./notebook/lambda_function.py ./data/sample_img ${LAMBDA_TASK_ROOT}

# Ensure the lamda user can execute the code
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]