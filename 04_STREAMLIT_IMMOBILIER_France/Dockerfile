# Start with a lightweight Linux Anaconda image
FROM continuumio/miniconda3

# Update all packages and install nano unzip and curl
RUN apt-get update
RUN apt-get install nano unzip curl -y

# THIS IS SPECIFIC TO HUGGINGFACE
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ---- Installer les dépendances Python via requirements.txt ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local files (avec l'utilisateur "user" comme propriétaire)
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD streamlit run --server.port 7860 app.py