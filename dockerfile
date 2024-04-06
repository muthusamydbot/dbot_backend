   # base python image for custom image
FROM python:3.9.6-slim-buster

# create working directory and install pip dependencies
WORKDIR /usr/dev/
COPY . .
RUN pip install -r requirements.txt

# copy python project files from local to /hello-py image working directory
COPY . .

# run the flask server 
CMD ["python3", "main.py"]