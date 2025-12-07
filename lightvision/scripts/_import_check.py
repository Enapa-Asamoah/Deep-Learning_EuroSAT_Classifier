import sys
import os
import traceback

# Add the project root (current dir) to sys.path so `src` can be found
sys.path.insert(0, os.path.abspath('.'))

try:
    import src
    print('OK', getattr(src, '__file__', getattr(src, '__path__', None)))
except Exception:
    traceback.print_exc()
    sys.exit(1)
