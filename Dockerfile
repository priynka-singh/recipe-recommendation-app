FROM python:3.11-bookworm

ENV PYTHONUNBUFFERED True

ENV APP_HOME /back-end

WORKDIR $APP_HOME

# Copy the application code into the Docker image
COPY . ./

# Copy the dataset into the Docker image
COPY final_dataset.csv $APP_HOME/

# Install NLTK and download stopwords and punkt during Docker build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install nltk

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]

# Copy the NLTK data directory to the expected location
RUN cp -r /root/nltk_data /usr/local/share/nltk_data

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
