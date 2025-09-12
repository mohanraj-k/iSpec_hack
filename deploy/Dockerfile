# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Flask/Gunicorn will run on
EXPOSE 5050

# Start Gunicorn
CMD ["gunicorn", "-c", "deploy/gunicorn.conf.py", "app:app"]
