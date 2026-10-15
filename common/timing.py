import time, tracemalloc
from contextlib import contextmanager


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


@contextmanager
def mem_tracker(enabled: bool, out: dict):
    if not enabled:
        yield
        return
    tracemalloc.start()
    try:
        yield
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["peak_mem_mb"] = peak / 1e6
