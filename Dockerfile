# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install the dependencies
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Copy the rest of the application code into the container
# COPY . .

# Set the environment variable for the application
ENV APP_ENV=production

# Expose the application port
EXPOSE 5000

# Set the command to run the application
CMD ["requirements","python", "resnet.py"] 