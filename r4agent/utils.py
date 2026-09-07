from dotenv import load_dotenv
from rich.logging import RichHandler
from pathlib import Path
import logging
from functools import wraps
from time import perf_counter
from typing import Any, Callable
import sys

def setup_logging(force: bool = False):
    logging.basicConfig(
        force=force,
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            enable_link_path=True
        )]
    )
    
def read_env(path:Path):
    load_dotenv(dotenv_path=path)
    logging.info(f"Env değişkenleri içeri aktarıldı.")



def log_execution_time(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()

        try:
            return function(*args, **kwargs)
        finally:
            elapsed_time = perf_counter() - start_time
            instance = args[0] if args else None
            owner_name = instance.__class__.__name__ if instance is not None else function.__name__
            message = (
                f"[magenta]{owner_name}.{function.__name__}[/magenta] "
                f"[{elapsed_time:.3f}]"
            )
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.info(message)
            else:
                print(f"INFO     {message}", file=sys.stderr)

    return wrapper