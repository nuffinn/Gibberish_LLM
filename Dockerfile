FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler code
COPY handler.py .

# Start the handler
CMD [ "python", "-u", "handler.py" ] 