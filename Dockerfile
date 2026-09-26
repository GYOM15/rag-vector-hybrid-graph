# Image of the Streamlit RAG comparison app.
FROM python:3.11-slim

WORKDIR /app

# Install the package + its dependencies first (layer cached as long as
# pyproject.toml and src/ don't change). The Graph stack needs the spaCy
# NER model, which pip does not pull in on its own.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir . \
 && python -m spacy download en_core_web_sm

# Application code and evaluation data.
COPY app/ ./app/
COPY eval/ ./eval/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
