import time
from threading import Thread

import psutil
import torch

MONITOR_INTERVAL = 30


class ResourceMonitor:
    def __init__(self, interval=MONITOR_INTERVAL):
        self.interval = interval
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def _monitor(self):
        while self.running:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            print("\n=== Resource Monitor ===")
            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory Used: {memory.percent}%")
            print(f"Available Memory: {memory.available / (1024 * 1024 * 1024):.2f} GB")

            if torch.backends.mps.is_available():
                print("MPS/GPU is active")

            print("=====================")
            time.sleep(self.interval)
