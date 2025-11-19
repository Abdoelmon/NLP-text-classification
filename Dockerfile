FROM python:3.9-slim


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/app/nltk_data 
RUN mkdir -p /app/nltk_data 
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('wordnet', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data');nltk.download('punkt_tab', download_dir='/app/nltk_data')"

COPY linear_svc_new_model.joblib .
COPY count_ngram_vectorizer.joblib .

COPY . .




EXPOSE 8501


RUN chmod +x /app/start.sh


CMD ["/app/start.sh"]