web: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT web_app:app
release: echo "Ensuring only web process is running - no worker processes needed"