FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DESKSENSE_RUN=/models/run DESKSENSE_DEVICE=cpu
WORKDIR /app
COPY pyproject.toml ./
COPY desksense ./desksense
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "desksense.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
