import os
import sys

# Add the parent directory to sys.path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        config.DATA_DIR,
        config.STUDENTS_DIR,
        config.LOGS_DIR,
        config.SRC_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Ensured directory exists: {directory}")

if __name__ == "__main__":
    ensure_directories()
