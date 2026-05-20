"""
logger.py  ── 模块三：打卡数据记录器

功能：
  - 将识别结果写入 CSV（姓名、日期、时间）
  - 防抖：同一人在 cooldown_minutes 内只记录一次
  - 线程安全（attendance.py 主循环与 IO 同进程即可）
"""

import csv
import os
import threading
from datetime import datetime, timedelta


class AttendanceLogger:
    """
    考勤日志记录器（线程安全，带防抖去重）。

    Parameters
    ----------
    log_file        : CSV 文件路径，默认 './logs/attendance_log.csv'
    cooldown_minutes: 同一人两次打卡最小间隔（分钟），默认 5
    """

    CSV_HEADER = ["姓名", "日期", "打卡时间"]

    def __init__(
        self,
        log_file: str = "./logs/attendance_log.csv",
        cooldown_minutes: int = 5,
    ):
        self.log_file = log_file
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._lock = threading.Lock()
        # {name: last_logged_datetime}
        self._last_seen: dict[str, datetime] = {}
        self._ensure_file()

    # ── 公开接口 ────────────────────────────────

    def log(self, name: str) -> bool:
        """
        尝试记录一次打卡事件。

        Parameters
        ----------
        name : 已识别的员工姓名（"Unknown" 会被自动忽略）

        Returns
        -------
        True  → 成功写入 CSV
        False → 在防抖冷却期内，跳过
        """
        if name == "Unknown":
            return False

        now = datetime.now()
        with self._lock:
            last = self._last_seen.get(name)
            if last and (now - last) < self._cooldown:
                return False

            self._last_seen[name] = now
            self._write_row(name, now)
            return True

    def today_summary(self) -> list[dict]:
        """返回今日所有打卡记录（列表，每条为 dict）"""
        today = datetime.now().strftime("%Y-%m-%d")
        records = []
        with self._lock:
            try:
                with open(self.log_file, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("日期") == today:
                            records.append(dict(row))
            except FileNotFoundError:
                pass
        return records

    def reset_cooldown(self, name: str | None = None) -> None:
        """
        手动重置防抖状态（测试用）。
        name=None 时清除所有人的状态。
        """
        with self._lock:
            if name:
                self._last_seen.pop(name, None)
            else:
                self._last_seen.clear()

    # ── 内部方法 ────────────────────────────────

    def _ensure_file(self) -> None:
        """确保 CSV 文件存在且含有表头"""
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)

    def _write_row(self, name: str, dt: datetime) -> None:
        """向 CSV 追加一行记录"""
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
            ])
        print(f"[Logger] ✅ 打卡记录：{name}  {dt.strftime('%Y-%m-%d %H:%M:%S')}")
