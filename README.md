====================================================================
PROJECT: TRUE/FAKE NEWS CLASSIFICATION (NLP Standalone Deployment)
====================================================================

This file provides the essential documentation and instructions for running the NLP classification project. The application classifies English text as True or Fake using a pre-trained machine learning model deployed via a standalone Streamlit application.

### CORE ARCHITECTURE:
* **Application:** Standalone Streamlit App.
* **API:** Fast api .

* **Classifier:** Linear Support Vector Classifier (Linear SVC).
* **Feature Extraction:** CountVectorizer (N-Grams).

====================================================================
1. PROJECT STRUCTURE (FLAT)
====================================================================
All files MUST reside in the same root directory:

├── deploy_app.py  (Main Streamlit application code)
├── app.py           (fast api code)
├── utils.py           (Preprocessing functions)
├── linear_svc_model.joblib        (Trained Model)
├── tfidf_ngram_vectorizer.joblib  (Fitted CountVectorizer object)
└── requirements.txt

====================================================================
2. SETUP AND INSTALLATION
====================================================================

1.  CREATE & ACTIVATE VIRTUAL ENVIRONMENT:
    python -m venv venv
    .\venv\Scripts\activate.bat   # Windows example
    
2.  INSTALL DEPENDENCIES:
    pip install -r requirements.txt

3.  DOWNLOAD NLTK RESOURCES (CRITICAL):
    The preprocessing utility requires NLTK data:
    python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])"

4.  PLACE ARTIFACTS:
    Ensure the two .joblib files (the model and the vectorizer) are in the root directory.

====================================================================
3. EXECUTION
====================================================================

Once the environment is set up and activated, run the Streamlit application directly:

streamlit run deploy_app.py

for fast api run 
uvicorn app:app --reload
====================================================================
4. DOCKERIZED EXECUTION (ISOLATED SERVICES)
====================================================================

This configuration uses a start script (start.sh) to isolate the services, allowing you to run only one service per container (API or Web).

### A. Build the Docker Image
docker build -t nlp-classification-service .

### B. Run FastAPI (API Backend)
Starts the API service on port 8000. Use this if you want an external service to access the model.

docker run -d \
    -p 8000:8000 \
    --name nlp_api_container \
    nlp-classification-service api

### C. Run Streamlit (Web UI)
Starts the interactive web interface on port 8501 (uses the standalone model within the container).

docker run -d \
    -p 8501:8501 \
    --name nlp_web_container \
    nlp-classification-service web

*Access the Web App: http://localhost:8501*
====================================================================
5. TROUBLESHOOTING COMMON ERRORS
====================================================================

* **FileNotFoundError:** Check the paths. Ensure the .joblib files are in the SAME folder as standalone_app.py.
* **NotFittedError / ValueError:** This means the model (LinearSVC) and the vectorizer are incompatible (trained on different feature counts). You must re-run your training script to generate new, compatible .joblib files.