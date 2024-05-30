import logging
import logger
import os
from datetime import datetime

# Generate the log file name with current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
print(LOG_FILE)

# Create the logs directory path
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create the full path for the log file
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging started")
