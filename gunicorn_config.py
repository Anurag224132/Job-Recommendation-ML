# Gunicorn configuration file for production deployment
import multiprocessing

# Bind to all interfaces on dynamic port (Render compatibility), fallback to 5001

import os
port = os.environ.get("PORT", "5001")
bind = f"0.0.0.0:{port}"

# Workers calculation: (2 * number of cores) + 1
# This is a standard recommendation for CPU-bound tasks
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class - 'sync' is default, for ML tasks where each request is CPU-heavy, 
# 'gthread' can be more efficient if using a thread-pool
worker_class = 'gthread'
threads = 4

# Timeout - increased for heavy PDF parsing and batch matching
timeout = 120

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout
loglevel = "info"

# Preload application to share memory between workers
preload_app = True
