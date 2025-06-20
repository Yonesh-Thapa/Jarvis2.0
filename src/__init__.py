# This file marks the src directory as a Python package.

# Global debug flag used across modules. Disabled by default.
DEBUG = False

def debug_print(*args, **kwargs):
    """Utility wrapper for debug output respecting the DEBUG flag."""
    if DEBUG:
        print(*args, **kwargs)

