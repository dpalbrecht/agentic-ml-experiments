FROM python:3.11-slim

WORKDIR /project

# System dependencies (libgomp for OpenMP, used by implicit library)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Start with common ML basics — commands will pip install more as needed
# and requirements.txt gets updated to track what's installed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project gets mounted at /project at runtime
