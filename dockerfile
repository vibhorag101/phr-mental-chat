# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install a suitable C compiler (in this case, gcc)
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app/

# Install dependencies using pip
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that the app runs on
EXPOSE 5555

# Start FastAPI app and Celery worker when the container launches
CMD ["python", "run.py"]