FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# We copy the requirements for the project
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# We copy the rest of the project
COPY main.py .

# Expose the streamlit port
EXPOSE 8501

# Command to execute the app
CMD ["streamlit", "run", "main.py"]

