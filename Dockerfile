# Pull official base image from Docker Hub
FROM python:3.11.5-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files into the container
COPY . .

# Expose the port the app runs on (e.g., 8000 for development, or a Gunicorn port for production)
EXPOSE 8000

# Command to run the application (for development)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
