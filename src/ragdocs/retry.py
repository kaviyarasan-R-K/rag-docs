import time

RETRYABLE = ["403", "RESOURCE_EXHAUSTED", "503", "INTERNAL SERVER ERROR", "UNAVAILABLE"]

def with_retry(fn, tries: int=5, base: float=2.0, sleep=time.sleep):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            last_try = attempt == tries -1
            if last_try or not any(token in str(exc) for token in RETRYABLE):
                raise
            sleep(base**attempt)

