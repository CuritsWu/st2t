from collections import deque
from threading import Lock
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class SimpleThreadDeque(deque[T]):
    """
    一個輕量級、執行緒安全的 deque（具 maxlen）
    - 繼承 collections.deque，所以原生 API 都可用
    - 於 append/appendleft/pop/popleft/extend... 自動加 Lock
    - 其餘唯讀操作如 __len__、__iter__ 也包成原子區段

    使用範例
    --------
    buf = SimpleThreadDeque(maxlen=50)
    buf.append(frame)
    audio = buf.popleft()
    """

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        maxlen: Optional[int] = None,
    ):
        super().__init__(iterable or (), maxlen)
        self._lock: Lock = Lock()

    # ----------- 主要寫入動作（丟最舊留最新） -----------
    def append(self, item: T) -> None:  # type: ignore[override]
        with self._lock:
            super().append(item)

    def appendleft(self, item: T) -> None:  # type: ignore[override]
        with self._lock:
            super().appendleft(item)

    def extend(self, items: Iterable[T]) -> None:  # type: ignore[override]
        with self._lock:
            super().extend(items)

    def extendleft(self, items: Iterable[T]) -> None:  # type: ignore[override]
        with self._lock:
            super().extendleft(items)

    # ----------- 讀出／移除動作 -----------
    def pop(self) -> T:  # type: ignore[override]
        with self._lock:
            return super().pop()

    def popleft(self) -> T:  # type: ignore[override]
        with self._lock:
            return super().popleft()

    def clear(self) -> None:  # type: ignore[override]
        with self._lock:
            super().clear()

    # ----------- 只讀屬性 -----------
    def __len__(self) -> int:  # type: ignore[override]
        with self._lock:
            return super().__len__()

    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        with self._lock:
            return iter(list(self))  # 複製一份避免遍歷時被改動

    # 其他還想鎖住的方法，可依需要覆寫


# --------------------- 測試用例 ---------------------
if __name__ == "__main__":
    import random
    import threading
    import time

    dq = SimpleThreadDeque(maxlen=10)

    def producer():
        for i in range(30):
            dq.append(i)
            print(f"P → {i:02d}  (len={len(dq)})")
            time.sleep(random.random() * 0.05)

    def consumer():
        while True:
            try:
                item = dq.popleft()
                print(f"      C ← {item:02d} (len={len(dq)})")
            except IndexError:
                pass
            time.sleep(0.03)

    threading.Thread(target=producer, daemon=True).start()
    threading.Thread(target=consumer, daemon=True).start()
    time.sleep(2)
