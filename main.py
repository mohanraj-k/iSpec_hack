import logging
# Set root logger to INFO so user logger.info messages appear
# logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)

# Configure logging to write to Gunicorn's error log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.FileHandler("gunicorn_error.log", mode="w"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Remove all handlers from root logger except FileHandler to silence terminal output
#Comment this block to print in Terminal
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if not isinstance(handler, logging.FileHandler):
        root_logger.removeHandler(handler)
##############################################

from app import app
from utils.config import DEBUG, PORT

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
