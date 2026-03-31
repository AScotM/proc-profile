#!/usr/bin/env python3

import os
import sys
import time
import json
import math
import signal
import argparse
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf(os.sysconf_names["SC_PAGE_SIZE"])
RUNNING = True


@dataclass
class ProcessSample:
    pid: int
    ppid: int
    comm: str
    exe_path: str
    state: str
    start_ticks: int
    start_time_epoch: float
    utime_ticks: int
    stime_ticks: int
    total_cpu_ticks: int
    rss_kb: int
    threads: int
    fd_count: int
    socket_fd_count: int
    cmdline: str


@dataclass
class ProcessBaseline:
    key: str
    samples: int = 0
    cpu_mean: float = 0.0
    cpu_m2: float = 0.0
    rss_mean: float = 0.0
    rss_m2: float = 0.0
    threads_mean: float = 0.0
    threads_m2: float = 0.0
    fd_mean: float = 0.0
    fd_m2: float = 0.0
    socket_fd_mean: float = 0.0
    socket_fd_m2: float = 0.0
    last_seen: float = 0.0
    comm: str = ""
    exe_path: str = ""

    def update_metric(self, value: float, mean_attr: str, m2_attr: str) -> None:
        count = self.samples
        mean = getattr(self, mean_attr)
        m2 = getattr(self, m2_attr)

        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2

        setattr(self, mean_attr, mean)
        setattr(self, m2_attr, m2)

    def update(self, metrics: "ProcessMetrics", seen_at: float) -> None:
        self.samples += 1
        self.update_metric(metrics.cpu_percent, "cpu_mean", "cpu_m2")
        self.update_metric(float(metrics.rss_kb), "rss_mean", "rss_m2")
        self.update_metric(float(metrics.threads), "threads_mean", "threads_m2")
        self.update_metric(float(metrics.fd_count), "fd_mean", "fd_m2")
        self.update_metric(float(metrics.socket_fd_count), "socket_fd_mean", "socket_fd_m2")
        self.last_seen = seen_at

    def stddev(self, mean_attr: str, m2_attr: str) -> float:
        if self.samples < 2:
            return 0.0
        return math.sqrt(getattr(self, m2_attr) / (self.samples - 1))

    def cpu_stddev(self) -> float:
        return self.stddev("cpu_mean", "cpu_m2")

    def rss_stddev(self) -> float:
        return self.stddev("rss_mean", "rss_m2")

    def threads_stddev(self) -> float:
        return self.stddev("threads_mean", "threads_m2")

    def fd_stddev(self) -> float:
        return self.stddev("fd_mean", "fd_m2")

    def socket_fd_stddev(self) -> float:
        return self.stddev("socket_fd_mean", "socket_fd_m2")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProcessBaseline":
        return cls(
            key=str(data.get("key", "")),
            samples=int(data.get("samples", 0)),
            cpu_mean=float(data.get("cpu_mean", 0.0)),
            cpu_m2=float(data.get("cpu_m2", 0.0)),
            rss_mean=float(data.get("rss_mean", 0.0)),
            rss_m2=float(data.get("rss_m2", 0.0)),
            threads_mean=float(data.get("threads_mean", 0.0)),
            threads_m2=float(data.get("threads_m2", 0.0)),
            fd_mean=float(data.get("fd_mean", 0.0)),
            fd_m2=float(data.get("fd_m2", 0.0)),
            socket_fd_mean=float(data.get("socket_fd_mean", 0.0)),
            socket_fd_m2=float(data.get("socket_fd_m2", 0.0)),
            last_seen=float(data.get("last_seen", 0.0)),
            comm=str(data.get("comm", "")),
            exe_path=str(data.get("exe_path", "")),
        )


@dataclass
class ProcessMetrics:
    key: str
    pid: int
    ppid: int
    comm: str
    exe_path: str
    state: str
    cmdline: str
    cpu_percent: float
    rss_kb: int
    threads: int
    fd_count: int
    socket_fd_count: int
    start_time_epoch: float
    start_ticks: int


@dataclass
class MetricAnomaly:
    metric: str
    value: float
    mean: float
    stddev: float
    zscore: float


@dataclass
class ProcessAlert:
    key: str
    pid: int
    comm: str
    exe_path: str
    severity: str
    score: float
    anomalies: List[MetricAnomaly] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    cpu_percent: float = 0.0
    rss_kb: int = 0
    threads: int = 0
    fd_count: int = 0
    socket_fd_count: int = 0
    cmdline: str = ""

    def to_line(self) -> str:
        metric_bits = []
        for item in self.anomalies:
            metric_bits.append(
                f"{item.metric}={item.value:.2f} mean={item.mean:.2f} std={item.stddev:.2f} z={item.zscore:.2f}"
            )
        metrics_text = "; ".join(metric_bits) if metric_bits else "no metric anomalies"
        reasons_text = "; ".join(self.reasons) if self.reasons else ""
        detail = f"{metrics_text}"
        if reasons_text:
            detail = f"{detail}; {reasons_text}"
        return (
            f"[{self.severity}] pid={self.pid} comm={self.comm} "
            f"exe={self.exe_path or '-'} score={self.score:.2f} {detail}"
        )


def install_signal_handlers() -> None:
    def handler(signum: int, frame: object) -> None:
        global RUNNING
        RUNNING = False

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def safe_readlink(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError:
        return ""


def parse_proc_stat(pid: int) -> Tuple[str, int, int, str, int, int, int]:
    path = f"/proc/{pid}/stat"
    raw = read_text(path).strip()

    left = raw.find("(")
    right = raw.rfind(")")
    if left == -1 or right == -1 or right <= left:
        raise ValueError(f"Malformed stat for pid {pid}")

    comm = raw[left + 1:right]
    rest = raw[right + 2:].split()

    state = rest[0]
    ppid = int(rest[1])
    utime_ticks = int(rest[11])
    stime_ticks = int(rest[12])
    threads = int(rest[17])
    start_ticks = int(rest[19])
    rss_pages = int(rest[21])

    return comm, ppid, utime_ticks, state, stime_ticks, threads, start_ticks, rss_pages


def parse_proc_status(pid: int) -> Dict[str, str]:
    data: Dict[str, str] = {}
    path = f"/proc/{pid}/status"
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def parse_proc_cmdline(pid: int) -> str:
    path = f"/proc/{pid}/cmdline"
    try:
        raw = open(path, "rb").read()
    except OSError:
        return ""
    parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]
    return " ".join(parts)


def count_fds_and_sockets(pid: int) -> Tuple[int, int]:
    fd_dir = f"/proc/{pid}/fd"
    fd_count = 0
    socket_fd_count = 0

    try:
        for entry in os.scandir(fd_dir):
            fd_count += 1
            try:
                target = os.readlink(entry.path)
                if target.startswith("socket:["):
                    socket_fd_count += 1
            except OSError:
                pass
    except OSError:
        return 0, 0

    return fd_count, socket_fd_count


def get_boot_time_epoch() -> float:
    for line in read_text("/proc/stat").splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    raise RuntimeError("Could not determine boot time")


def list_pids() -> List[int]:
    pids = []
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            pids.append(int(entry.name))
    return pids


class ProcFingerprinter:
    def __init__(
        self,
        interval: float = 2.0,
        warmup_samples: int = 5,
        z_threshold: float = 3.0,
        min_abs_cpu: float = 10.0,
        min_abs_rss_kb: int = 10240,
        min_abs_threads: int = 5,
        min_abs_fds: int = 10,
        min_abs_socket_fds: int = 5,
        baseline_path: Optional[str] = None,
        top_n: int = 20,
        include_cmdline: bool = False,
    ) -> None:
        self.interval = interval
        self.warmup_samples = warmup_samples
        self.z_threshold = z_threshold
        self.min_abs_cpu = min_abs_cpu
        self.min_abs_rss_kb = min_abs_rss_kb
        self.min_abs_threads = min_abs_threads
        self.min_abs_fds = min_abs_fds
        self.min_abs_socket_fds = min_abs_socket_fds
        self.baseline_path = baseline_path
        self.top_n = top_n
        self.include_cmdline = include_cmdline
        self.boot_time_epoch = get_boot_time_epoch()
        self.baselines: Dict[str, ProcessBaseline] = {}

    def make_key(self, exe_path: str, comm: str) -> str:
        left = exe_path if exe_path else "-"
        right = comm if comm else "-"
        return f"{left}|{right}"

    def collect_samples(self) -> Dict[int, ProcessSample]:
        now = time.time()
        samples: Dict[int, ProcessSample] = {}

        for pid in list_pids():
            try:
                comm, ppid, utime_ticks, state, stime_ticks, threads, start_ticks, rss_pages = parse_proc_stat(pid)
                exe_path = safe_readlink(f"/proc/{pid}/exe")
                fd_count, socket_fd_count = count_fds_and_sockets(pid)
                cmdline = parse_proc_cmdline(pid) if self.include_cmdline else ""
                rss_kb = (rss_pages * PAGE_SIZE) // 1024
                start_time_epoch = self.boot_time_epoch + (start_ticks / CLK_TCK)

                samples[pid] = ProcessSample(
                    pid=pid,
                    ppid=ppid,
                    comm=comm,
                    exe_path=exe_path,
                    state=state,
                    start_ticks=start_ticks,
                    start_time_epoch=start_time_epoch,
                    utime_ticks=utime_ticks,
                    stime_ticks=stime_ticks,
                    total_cpu_ticks=utime_ticks + stime_ticks,
                    rss_kb=rss_kb,
                    threads=threads,
                    fd_count=fd_count,
                    socket_fd_count=socket_fd_count,
                    cmdline=cmdline,
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
                continue

        return samples

    def build_metrics(
        self,
        previous: Dict[int, ProcessSample],
        current: Dict[int, ProcessSample],
        elapsed: float,
    ) -> List[ProcessMetrics]:
        metrics: List[ProcessMetrics] = []

        if elapsed <= 0:
            return metrics

        cpu_denominator = elapsed * CLK_TCK

        for pid, curr in current.items():
            prev = previous.get(pid)
            if prev is None:
                continue

            if prev.start_ticks != curr.start_ticks:
                continue

            cpu_delta_ticks = curr.total_cpu_ticks - prev.total_cpu_ticks
            cpu_percent = (cpu_delta_ticks / cpu_denominator) * 100.0
            if cpu_percent < 0:
                cpu_percent = 0.0

            key = self.make_key(curr.exe_path, curr.comm)

            metrics.append(
                ProcessMetrics(
                    key=key,
                    pid=curr.pid,
                    ppid=curr.ppid,
                    comm=curr.comm,
                    exe_path=curr.exe_path,
                    state=curr.state,
                    cmdline=curr.cmdline,
                    cpu_percent=cpu_percent,
                    rss_kb=curr.rss_kb,
                    threads=curr.threads,
                    fd_count=curr.fd_count,
                    socket_fd_count=curr.socket_fd_count,
                    start_time_epoch=curr.start_time_epoch,
                    start_ticks=curr.start_ticks,
                )
            )

        return metrics

    def get_or_create_baseline(self, metrics: ProcessMetrics) -> ProcessBaseline:
        baseline = self.baselines.get(metrics.key)
        if baseline is None:
            baseline = ProcessBaseline(
                key=metrics.key,
                comm=metrics.comm,
                exe_path=metrics.exe_path,
            )
            self.baselines[metrics.key] = baseline
        return baseline

    def metric_anomaly(
        self,
        name: str,
        value: float,
        mean: float,
        stddev: float,
        z_threshold: float,
        abs_threshold: float,
    ) -> Optional[MetricAnomaly]:
        if stddev <= 0:
            return None

        zscore = (value - mean) / stddev
        if zscore < z_threshold:
            return None

        if value < abs_threshold:
            return None

        return MetricAnomaly(
            metric=name,
            value=value,
            mean=mean,
            stddev=stddev,
            zscore=zscore,
        )

    def score_alert(self, anomalies: List[MetricAnomaly], reasons: List[str]) -> Tuple[float, str]:
        score = 0.0

        for item in anomalies:
            score += max(0.0, item.zscore - self.z_threshold + 1.0)

        score += float(len(reasons))

        if score >= 12.0:
            severity = "CRITICAL"
        elif score >= 7.0:
            severity = "HIGH"
        elif score >= 3.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return score, severity

    def detect_alert(self, metrics: ProcessMetrics, baseline: ProcessBaseline) -> Optional[ProcessAlert]:
        if baseline.samples < self.warmup_samples:
            return None

        anomalies: List[MetricAnomaly] = []
        reasons: List[str] = []

        cpu_anomaly = self.metric_anomaly(
            "cpu_percent",
            metrics.cpu_percent,
            baseline.cpu_mean,
            baseline.cpu_stddev(),
            self.z_threshold,
            self.min_abs_cpu,
        )
        if cpu_anomaly:
            anomalies.append(cpu_anomaly)

        rss_anomaly = self.metric_anomaly(
            "rss_kb",
            float(metrics.rss_kb),
            baseline.rss_mean,
            baseline.rss_stddev(),
            self.z_threshold,
            float(self.min_abs_rss_kb),
        )
        if rss_anomaly:
            anomalies.append(rss_anomaly)

        threads_anomaly = self.metric_anomaly(
            "threads",
            float(metrics.threads),
            baseline.threads_mean,
            baseline.threads_stddev(),
            self.z_threshold,
            float(self.min_abs_threads),
        )
        if threads_anomaly:
            anomalies.append(threads_anomaly)

        fd_anomaly = self.metric_anomaly(
            "fd_count",
            float(metrics.fd_count),
            baseline.fd_mean,
            baseline.fd_stddev(),
            self.z_threshold,
            float(self.min_abs_fds),
        )
        if fd_anomaly:
            anomalies.append(fd_anomaly)

        socket_fd_anomaly = self.metric_anomaly(
            "socket_fd_count",
            float(metrics.socket_fd_count),
            baseline.socket_fd_mean,
            baseline.socket_fd_stddev(),
            self.z_threshold,
            float(self.min_abs_socket_fds),
        )
        if socket_fd_anomaly:
            anomalies.append(socket_fd_anomaly)

        if baseline.exe_path and metrics.exe_path and baseline.exe_path != metrics.exe_path:
            reasons.append("exe path changed")

        if metrics.state == "Z":
            reasons.append("process is zombie")

        if metrics.socket_fd_count > metrics.fd_count and metrics.fd_count > 0:
            reasons.append("socket fd count exceeds fd count")

        if not anomalies and not reasons:
            return None

        score, severity = self.score_alert(anomalies, reasons)

        return ProcessAlert(
            key=metrics.key,
            pid=metrics.pid,
            comm=metrics.comm,
            exe_path=metrics.exe_path,
            severity=severity,
            score=score,
            anomalies=anomalies,
            reasons=reasons,
            cpu_percent=metrics.cpu_percent,
            rss_kb=metrics.rss_kb,
            threads=metrics.threads,
            fd_count=metrics.fd_count,
            socket_fd_count=metrics.socket_fd_count,
            cmdline=metrics.cmdline,
        )

    def update_baselines(self, metrics_list: List[ProcessMetrics], seen_at: float) -> None:
        for metrics in metrics_list:
            baseline = self.get_or_create_baseline(metrics)
            baseline.comm = metrics.comm
            baseline.exe_path = metrics.exe_path
            baseline.update(metrics, seen_at)

    def analyze_metrics(self, metrics_list: List[ProcessMetrics]) -> List[ProcessAlert]:
        alerts: List[ProcessAlert] = []
        for metrics in metrics_list:
            baseline = self.get_or_create_baseline(metrics)
            alert = self.detect_alert(metrics, baseline)
            if alert is not None:
                alerts.append(alert)
        alerts.sort(key=lambda item: item.score, reverse=True)
        return alerts[: self.top_n]

    def save_baselines(self) -> None:
        if not self.baseline_path:
            return

        data = {
            "saved_at": time.time(),
            "boot_time_epoch": self.boot_time_epoch,
            "baselines": {key: baseline.to_dict() for key, baseline in self.baselines.items()},
        }

        tmp_path = f"{self.baseline_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.baseline_path)

    def load_baselines(self) -> None:
        if not self.baseline_path:
            return

        try:
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            return

        baselines_raw = data.get("baselines", {})
        if not isinstance(baselines_raw, dict):
            return

        loaded: Dict[str, ProcessBaseline] = {}
        for key, item in baselines_raw.items():
            if isinstance(item, dict):
                baseline = ProcessBaseline.from_dict(item)
                loaded[str(key)] = baseline

        self.baselines = loaded

    def print_snapshot_summary(self, metrics_list: List[ProcessMetrics]) -> None:
        if not metrics_list:
            print("No process metrics collected")
            return

        cpu_values = [m.cpu_percent for m in metrics_list]
        rss_values = [m.rss_kb for m in metrics_list]
        thread_values = [m.threads for m in metrics_list]
        fd_values = [m.fd_count for m in metrics_list]
        socket_fd_values = [m.socket_fd_count for m in metrics_list]

        print(
            "summary "
            f"processes={len(metrics_list)} "
            f"cpu_mean={statistics.fmean(cpu_values):.2f} "
            f"rss_mean_kb={statistics.fmean(rss_values):.2f} "
            f"threads_mean={statistics.fmean(thread_values):.2f} "
            f"fd_mean={statistics.fmean(fd_values):.2f} "
            f"socket_fd_mean={statistics.fmean(socket_fd_values):.2f}"
        )

    def print_top_processes(self, metrics_list: List[ProcessMetrics], sort_by: str = "cpu") -> None:
        if sort_by == "rss":
            ranked = sorted(metrics_list, key=lambda m: m.rss_kb, reverse=True)
        elif sort_by == "fds":
            ranked = sorted(metrics_list, key=lambda m: m.fd_count, reverse=True)
        elif sort_by == "threads":
            ranked = sorted(metrics_list, key=lambda m: m.threads, reverse=True)
        else:
            ranked = sorted(metrics_list, key=lambda m: m.cpu_percent, reverse=True)

        print(f"top_by_{sort_by}")
        print(
            f"{'pid':>7} {'cpu%':>8} {'rss_kb':>10} {'thr':>6} {'fds':>6} {'sock':>6} {'comm':<24} exe"
        )

        for item in ranked[: self.top_n]:
            print(
                f"{item.pid:>7} "
                f"{item.cpu_percent:>8.2f} "
                f"{item.rss_kb:>10} "
                f"{item.threads:>6} "
                f"{item.fd_count:>6} "
                f"{item.socket_fd_count:>6} "
                f"{item.comm:<24.24} "
                f"{item.exe_path}"
            )

    def print_alerts(self, alerts: List[ProcessAlert]) -> None:
        if not alerts:
            print("alerts none")
            return

        print("alerts")
        for alert in alerts:
            print(alert.to_line())
            if self.include_cmdline and alert.cmdline:
                print(f"  cmdline={alert.cmdline}")

    def run_once(self) -> int:
        first = self.collect_samples()
        start = time.time()
        time.sleep(self.interval)
        second = self.collect_samples()
        end = time.time()

        metrics_list = self.build_metrics(first, second, end - start)
        alerts = self.analyze_metrics(metrics_list)
        self.print_snapshot_summary(metrics_list)
        self.print_top_processes(metrics_list, sort_by="cpu")
        self.print_alerts(alerts)
        self.update_baselines(metrics_list, end)
        self.save_baselines()
        return 0

    def run_loop(self) -> int:
        previous = self.collect_samples()
        previous_time = time.time()

        while RUNNING:
            time.sleep(self.interval)
            current = self.collect_samples()
            current_time = time.time()

            metrics_list = self.build_metrics(previous, current, current_time - previous_time)
            alerts = self.analyze_metrics(metrics_list)

            print(f"\ncycle ts={int(current_time)}")
            self.print_snapshot_summary(metrics_list)
            self.print_top_processes(metrics_list, sort_by="cpu")
            self.print_alerts(alerts)

            self.update_baselines(metrics_list, current_time)
            self.save_baselines()

            previous = current
            previous_time = current_time

        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proc-signal")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--min-abs-cpu", type=float, default=10.0)
    parser.add_argument("--min-abs-rss-kb", type=int, default=10240)
    parser.add_argument("--min-abs-threads", type=int, default=5)
    parser.add_argument("--min-abs-fds", type=int, default=10)
    parser.add_argument("--min-abs-socket-fds", type=int, default=5)
    parser.add_argument("--baseline-file", default="")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cmdline", action="store_true")
    return parser


def main() -> int:
    install_signal_handlers()
    parser = build_parser()
    args = parser.parse_args()

    tool = ProcFingerprinter(
        interval=args.interval,
        warmup_samples=args.warmup_samples,
        z_threshold=args.z_threshold,
        min_abs_cpu=args.min_abs_cpu,
        min_abs_rss_kb=args.min_abs_rss_kb,
        min_abs_threads=args.min_abs_threads,
        min_abs_fds=args.min_abs_fds,
        min_abs_socket_fds=args.min_abs_socket_fds,
        baseline_path=args.baseline_file or None,
        top_n=args.top,
        include_cmdline=args.cmdline,
    )

    tool.load_baselines()

    if args.once:
        return tool.run_once()

    return tool.run_loop()


if __name__ == "__main__":
    sys.exit(main())
