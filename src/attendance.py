import os
import csv
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_directories

class AttendanceManager:
    def __init__(self):
        """Initialize the AttendanceManager."""
        ensure_directories()
        self.log_file = config.ATTENDANCE_LOG_FILE
        self.session_marked = set()
        
        # Ensure CSV has headers if it doesn't exist or is empty
        self._initialize_csv()

    def _initialize_csv(self):
        """Create the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
                
    def mark_attendance(self, name):
        """
        Mark attendance for a given name if not already marked in this session.
        Returns True if newly marked, False if already marked.
        """
        # Unknown humans shouldn't be marked
        if name == "Unknown":
            return False
            
        if name in self.session_marked:
            return False
            
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str, time_str])
                
            self.session_marked.add(name)
            print(f"[LOG] Marked attendance for: {name} at {time_str}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not write to attendance log: {e}")
            return False
            
    def get_session_attendance(self):
        """Return the set of names marked in the current session."""
        return self.session_marked
