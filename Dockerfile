# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /DockerMnist
WORKDIR /DockerMnist

# Copy the current directory contents into the container at /DockerMnist
COPY . /DockerMnist

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run flask_mnist.py
CMD ["python3", "MnistApp.py"]
