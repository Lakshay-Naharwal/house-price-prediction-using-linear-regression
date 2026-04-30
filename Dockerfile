FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port required by Hugging Face
EXPOSE 7860

# Run with Gunicorn on HF required port
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
