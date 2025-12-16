# Import modules
import time
from functools import wraps

def timing_decorator(func):
    """
    This function implements a profiler to monitor the execution time of a python function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Determine if the function is a class method or not
        bound_instance = args[0] if args else None
        class_name = None

        if bound_instance:
            class_name = type(bound_instance).__name__

        # Start the timer
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        # Stop the timer and compute the elapsed time
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if class_name:
            print(f"[{class_name}.{func.__name__}] Tempo di esecuzione: {elapsed:.6f} secondi")
        else:
            print(f"[{func.__name__}] Tempo di esecuzione: {elapsed:.6f} secondi")

        return result
    
    return wrapper
