# Hugging Face V2 Inference Service

This repository contains a FastAPI service that acts as a proxy between the Hugging Face models deployed on Truefoundry and the clients using the models. The service converts the input data to the V2 inference protocol format and forwards the request to the deployed model's endpoint.

## Getting Started

These instructions will help you set up and run the Hugging Face V2 Inference Service on your local machine.

### Prerequisites

- Python 3.7 or above
- pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hf-v2-inference-service.git
   ```

2. Navigate to the project directory:

   ```bash
   cd hf-v2-inference-service
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Start the service:

   ```bash
   uvicorn main:app --reload --hf_pipeline <pipeline_name> --model_deployed_url <model_url>
   ```
    Replace <pipeline_name> with the desired Hugging Face pipeline name (e.g., zero-shot-classification, object-detection, text-generation, token-classification) and <model_url> with the deployed endpoint URL provided by Truefoundry.

2.  Once the service is up and running, you can send POST requests to http://localhost:8000 to make predictions. The request body should follow the input format specified in the Hugging Face documentation for the respective pipeline.
