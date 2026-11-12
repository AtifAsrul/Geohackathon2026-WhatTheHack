"""Initialize app-local limits before importing NumPy, SciPy, or model libraries."""
from .runtime import configure_threads

configure_threads()
