# this is the python version I used to develop the project
FROM python:3.11.7-slim

# set work directory
WORKDIR /app

# copy contents to the container
COPY . /app

# install necessary libraries
RUN pip install --no-cache-dir -r requirements.txt

# install spacy model
RUN python -m spacy download en_core_web_sm

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "main.py"]