web: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT web_app:app
worker: python ai_worker_standalone.py