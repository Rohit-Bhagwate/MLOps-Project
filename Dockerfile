FROM python:3.8-slim
RUN pip install numpy==1.26.4 scikit-learn==1.2.1 joblib pandas flask
COPY app.py /app/
COPY models/* /app/models/ 
WORKDIR /app
CMD ["python", "app.py"]