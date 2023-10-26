# Start from a base image
FROM python:3.11

# Set the working directory
WORKDIR /APIfull

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["sub_grade_classifier.joblib","grade_classifier.joblib", "int_rate_regressor.joblib", "loan_classifier.joblib", "main.py", "functions.py", "classes.py","./"] .
#COPY ["__BUILD_NUMBER", "README.md", "gulpfile", "another_file", "./"]

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]