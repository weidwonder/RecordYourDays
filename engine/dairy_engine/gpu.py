"""选卡与显存采样。选卡逻辑是纯函数，便于单测；采样用 NVML，只在 GPU 方案的环境里导入。"""
import os
import subprocess
import threading


def parse_nvidia_smi(gpus_csv, apps_csv):
    """nvidia-smi 的两段 csv（noheader,nounits）→ [{index, uuid, used, total, apps}]。"""
    gpus = []
    for line in gpus_csv.strip().splitlines():
        idx, uuid, used, total = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "uuid": uuid, "used": int(used), "total": int(total), "apps": []})
    by_uuid = {g["uuid"]: g for g in gpus}
    for line in apps_csv.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        if uuid in by_uuid:
            by_uuid[uuid]["apps"].append({"pid": int(pid), "used": int(mem) if mem.isdigit() else None})
    return gpus


def pick_gpus(gpus, min_free_mib, allowed=None):
    """可用的卡：有其他进程在算（compute app）的不用；剩余显存不足 min_free_mib 的不用。allowed 限定候选编号。"""
    out = []
    for g in gpus:
        if allowed is not None and g["index"] not in allowed:
            continue
        if g["apps"] or g["total"] - g["used"] < min_free_mib:
            continue
        out.append(g["index"])
    return out


def query_gpus():
    q = lambda args: subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    return parse_nvidia_smi(q(["--query-gpu=index,uuid,memory.used,memory.total"]), q(["--query-compute-apps=gpu_uuid,pid,used_memory"]))


class VramMonitor:
    """每 100 ms 采样一次指定卡上属于本进程（及登记的子进程）的显存，记峰值 MiB。"""

    def __init__(self, index):
        self.index = index
        self.pids = {os.getpid()}
        self.peak = 0
        self.samples = 0
        self._stop = threading.Event()

    def __enter__(self):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                used = sum(p.usedGpuMemory or 0 for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
                           if p.pid in self.pids and (p.usedGpuMemory or 0) < 2**60)
                self.samples += 1
                self.peak = max(self.peak, used)
            except Exception:
                pass
            self._stop.wait(0.1)

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()

    @property
    def peak_mib(self):
        return round(self.peak / 2**20) if self.samples else None
