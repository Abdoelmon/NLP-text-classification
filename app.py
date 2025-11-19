from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import warnings
from utils import preprocess_text 


app = FastAPI(
    title="NLP Classification API (True/Fake)",
    description=" Linear SVC CountVectorizer. NLP Classification API for True vs Fake News Detection",
    version="1.0.0"
)


model = None
vectorizer = None


class TextPayload(BaseModel):
    text: str

@app.on_event("startup")
def load_model_artifacts():
    global model, vectorizer

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'linear_svc_new_model.joblib')
    vectorizer_path = os.path.join(base_dir, 'count_ngram_vectorizer.joblib') 

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
       
            if not hasattr(vectorizer, 'vocabulary_'):
                raise RuntimeError("Vectorizer is not fitted and cannot be used for transformation.")
                
        print("✅ Vectorizer loaded successfully.")

    except FileNotFoundError:
        print(f"❌ Model or vectorizer files not found.")
   
        model = None
        vectorizer = None
    except Exception as e:
        print(f"❌ error  {e}")
        model = None
        vectorizer = None

@app.post("/predict")
def predict_fake_news(payload: TextPayload):
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    try:
     
        cleaned_text = preprocess_text(payload.text)

       
    
        text_vector = vectorizer.transform([cleaned_text])

        
        prediction = model.predict(text_vector)[0]
        
        
        result = "Fake/مزيف" if prediction == 1 else "True/صحيح"

        return {
            "input_text": payload.text,
            "processed_text": cleaned_text,
            "prediction": int(prediction),
            "label": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/")
def read_root():
    return {"status": "FastAPI service is running", "model_loaded": model is not None}