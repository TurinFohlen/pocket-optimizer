from registry import registry
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

@registry.register(
    name='service.executor',
    type_='service',
    signature='executor_service()'   # 必须提供 signature
)
class ExecutorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_executors'):
            self._executors = {}

    def get_thread_pool(self, max_workers: int = 4):
        key = f'thread_{max_workers}'
        if key not in self._executors:
            self._executors[key] = ThreadPoolExecutor(max_workers=max_workers)
        return self._executors[key]

    def get_process_pool(self, max_workers: int = 4):
        key = f'process_{max_workers}'
        if key not in self._executors:
            self._executors[key] = ProcessPoolExecutor(max_workers=max_workers)
        return self._executors[key]

    def map(self, func, iterable, pool_type='thread', max_workers=4):
        if pool_type == 'thread':
            pool = self.get_thread_pool(max_workers)
        else:
            pool = self.get_process_pool(max_workers)
        return list(pool.map(func, iterable))

    def shutdown(self):
        for pool in self._executors.values():
            pool.shutdown(wait=False)
        self._executors.clear()
