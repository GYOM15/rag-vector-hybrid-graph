# Image de l'application Streamlit de comparaison RAG.
FROM python:3.11-slim

WORKDIR /app

# Installe d'abord le paquet + ses dépendances (couche mise en cache tant que
# pyproject.toml et src/ ne changent pas).
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Code applicatif et données d'évaluation.
COPY app/ ./app/
COPY eval/ ./eval/

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
