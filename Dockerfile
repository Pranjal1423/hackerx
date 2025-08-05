# ✅ Use Python 3.9
FROM python:3.9.13-slim

# Set working directory
WORKDIR /app

# Optional: Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Upgrade pip & build tools
RUN pip install --upgrade pip setuptools wheel build

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Start your FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
