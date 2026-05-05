import time
import os
import re

LOG_PATH = "/Users/bryan/.aura/logs/desktop-launch.log"
REPORT_PATH = "/Users/bryan/.aura/live-source/scratch/guardian_report.log"

PATTERNS = {
    "DEADLOCK": r"DEADLOCK|Deadlock",
    "ATTRIBUTE_ERROR": r"AttributeError",
    "SERVICE_ERROR": r"ServiceNotFoundError|Service '.*' not found",
    "STALL": r"Event loop blocked for (\d+\.\d+)s",
    "REPETITION": r"Token repetition detected|generative loop",
    "MOTOR_FAILURE": r"Motor cortex watchdog spuriously cancelled",
    "HEALTH_PULSE": r"UNIFIED HEALTH PULSE",
}

def monitor():
    print(f"Aura Guardian started. Monitoring {LOG_PATH}")
    last_pos = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    
    while True:
        if not os.path.exists(LOG_PATH):
            time.sleep(10)
            continue
            
        current_size = os.path.getsize(LOG_PATH)
        if current_size < last_pos: # Log rotated
            last_pos = 0
            
        if current_size > last_pos:
            with open(LOG_PATH, "r") as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()
                
                findings = []
                for key, pattern in PATTERNS.items():
                    matches = re.findall(pattern, new_content)
                    if matches:
                        findings.append(f"[{key}] Found {len(matches)} occurrences.")
                
                if findings:
                    with open(REPORT_PATH, "a") as report:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        report.write(f"\n--- {timestamp} ---\n")
                        report.write("\n".join(findings) + "\n")
        
        time.sleep(300) # Check every 5 minutes

if __name__ == "__main__":
    monitor()
