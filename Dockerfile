FROM python:3.13-slim

WORKDIR /app

# Install the package with server dependencies
COPY . .
RUN pip install --no-cache-dir ".[server]"

# Expose the default port
EXPOSE 8000

# Run the server
ENTRYPOINT ["llmling-models", "serve"]
CMD ["--auto-discover"]
