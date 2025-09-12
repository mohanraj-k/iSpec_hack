# Gunicorn configuration file
# Place this file in your project root and run gunicorn as usual (e.g., `gunicorn app:app`)
# These settings will be picked up automatically.

# Number of worker processes
workers = 1

# Bind to all interfaces on port 5050
bind = "0.0.0.0:5050"

# Worker timeout (in seconds)
timeout = 6000

# Log level
loglevel = "info"

# Access log file (requests)
accesslog = "gunicorn_access.log"

# Error log file (errors, tracebacks)
errorlog = "gunicorn_error.log"

# # (Optional) Access log file (uncomment to enable)
# accesslog = "-"  # log to stdout

# # (Optional) Error log file (uncomment to enable)
# errorlog = "-"   # log to stderr

# Capture stdout/stderr from workers so print() and other outputs are logged
capture_output = True