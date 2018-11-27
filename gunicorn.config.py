"""Gunicorn configuration."""

bind = '127.0.0.1:5001'

# 3223MiB GPU RAM by app
workers = 2
worker_class = 'gevent'

accesslog = '-'

