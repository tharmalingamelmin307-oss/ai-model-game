"""Helpers for keeping RKNN native startup logs out of the terminal."""

import contextlib
import os
import sys
import threading

import config


_OUTPUT_REDIRECT_LOCK = threading.Lock()
_RKNN_WARNING_FILTER_INSTALLED = False


_FILTER_PATTERNS = (
    "Query dynamic range failed",
    "query RKNN_QUERY_INPUT_DYNAMIC_RANGE error",
    "please export rknn with dynamic_shapes",
    "static shape RKNN model",
)


class _FilteredTextStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        if any(pattern in str(text) for pattern in _FILTER_PATTERNS):
            return len(text)
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def install_rknn_warning_filter():
    """Filter known harmless RKNN static-shape warning lines from Python streams."""
    global _RKNN_WARNING_FILTER_INSTALLED
    if _RKNN_WARNING_FILTER_INSTALLED:
        return
    if not bool(getattr(config, "SUPPRESS_RKNN_INIT_OUTPUT", True)):
        return

    sys.stdout = _FilteredTextStream(sys.stdout)
    sys.stderr = _FilteredTextStream(sys.stderr)
    _RKNN_WARNING_FILTER_INSTALLED = True


@contextlib.contextmanager
def suppress_rknn_init_output():
    """Temporarily silence native stdout/stderr during RKNN load/init."""
    if not bool(getattr(config, "SUPPRESS_RKNN_INIT_OUTPUT", True)):
        yield
        return

    with _OUTPUT_REDIRECT_LOCK:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    os.dup2(devnull.fileno(), 1)
                    os.dup2(devnull.fileno(), 2)
                    yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
