import logging 
import os
from datetime import datetime

# 1. Define the directory path where logs will be stored
LOG_DIR = os.path.join(os.getcwd(), "logs")

# 2. Define the filename using the current timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# 3. Create the log directory if it doesn't exist (using LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# 4. Define the full path to the log file by joining the directory and the filename
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)


logging.basicConfig(
    filename = LOG_FILE_PATH,  # Use the correctly constructed full path
    format= '[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s',
    level = logging.INFO,
)


# checking logging 

# if __name__ == "__main__":
#     logging.info("Logging has started.")


