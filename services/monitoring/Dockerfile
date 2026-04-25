FROM mlops-base:latest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
VOLUME ["/reports"]
CMD ["python", "monitor.py"]
