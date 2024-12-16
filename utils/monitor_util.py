import time
from threading import Thread

import psutil
import torch
import wandb
from torch.profiler import profile, ProfilerActivity
from transformers import TrainerCallback

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


class ProfilerCallback(TrainerCallback):
    def __init__(self, activities=[ProfilerActivity.GPU], record_shapes=True, profiler_steps=None):
        """
        初始化 ProfilerCallback
        :param activities: 要监控的活动类型，例如 [ProfilerActivity.CPU]
        :param record_shapes: 是否记录张量形状
        :param profiler_steps: 记录性能分析的训练步骤。例如，None 表示记录整个训练过程
        """
        self.activities = activities
        self.record_shapes = record_shapes
        self.profiler_steps = profiler_steps
        self.profiler = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # 初始化 profiler
        self.profiler = profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            schedule=torch.profiler.schedule(wait=4, warmup=4, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("log1"),
            profile_memory=True,  # 可选：是否记录内存信息
            with_stack=True  # 可选：是否记录调用栈
        )
        self.profiler.start()
        print("Profiler started.")

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()
            self.current_step += 1

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.stop()
            print("Profiler stopped.")

    def _wandb_trace_handler(self, p):
        """
            将 profiler trace 上传到 wandb
            :param p: profiler 对象
        """
        # 导出 trace 到本地文件
        trace_file = "trace.json"
        p.export_chrome_trace(trace_file)

        # 将 trace 文件作为 wandb Artifact 上传
        artifact = wandb.Artifact("profiler_trace", type="profiling")
        artifact.add_file(trace_file)
        wandb.log_artifact(artifact)

        print(f"Profiler trace saved and uploaded to wandb as {trace_file}.")
