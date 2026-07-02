FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY mcp_server ./mcp_server

EXPOSE 8000
EXPOSE 8001

# Default command runs the API; docker-compose overrides this for the
# mcp-server service to run `python -m mcp_server.server` instead -- same
# image, same dependencies, two different processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]