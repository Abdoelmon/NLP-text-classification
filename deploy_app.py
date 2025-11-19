import streamlit as st
import joblib
import os
from utils import preprocess_text
import numpy as np

st.set_page_config(page_title="News classification (True/Fake)", layout="wide")
st.title("💡 News Classifier - Standalone Streamlit App")
st.markdown("Uses a pre-trained Linear SVC model on your own data.")

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
   
    vectorizer_path = os.path.join(base_dir, 'count_ngram_vectorizer.joblib')
    model_path = os.path.join(base_dir, 'linear_svc_new_model.joblib')
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
       
        if not hasattr(vectorizer, 'vocabulary_'):
             st.error("❌  Vectorizer .")
             return None, None
        
        st.sidebar.success("✅ Model and Vectorizer loaded successfully.")
        return model, vectorizer
        
    except FileNotFoundError:
        st.error(f"❌ Error: Model files not found. Ensure .joblib files are in the same directory as the code.")
        return None, None
    except Exception as e:
        st.error(f"❌ Unexpected error during loading: {e}")
        return None, None

model, vectorizer = load_artifacts()

# 3. User Interaction and Classification Interface
if model and vectorizer:
    
    user_input = st.text_area("Enter the text to classify here:", 
                              value="The White House spokesperson denied the shocking rumors about the new economic policy.", 
                              height=150)
    
    if st.button("Predict Classification", type="primary"):
        if not user_input:
            st.warning("Please enter text first.")
            
        else:
            with st.spinner('Processing text and predicting...'):
                
                # a. Preprocessing the text
                cleaned_text = preprocess_text(user_input)
                
                # b. Feature transformation (using the trained vectorizer)
                text_vector = vectorizer.transform([cleaned_text])
                
                # c. Prediction (using the trained model)
                prediction = model.predict(text_vector)[0]
                
                # d. Displaying the result
                if prediction == 0:
                    result = "Fake/"
                    st.error(f"⚠️ Result: {result} (Likely Fake)")
                else:
                    result = "True"
                    st.success(f"✅ Result: {result} (Likely True)")
                    
                st.subheader("Data Analysis:")
                st.markdown(f"**Processed Text:** `{cleaned_text}`")