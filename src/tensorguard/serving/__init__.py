"""
TensorGuard Serving Package
"""

from .backend import TenSEALBackend, NativeBackend


def create_app():
    from .gateway import app

    return app


def start_server(*args, **kwargs):
    from .gateway import start_server as _start_server

    return _start_server(*args, **kwargs)


__all__ = ["TenSEALBackend", "NativeBackend", "create_app", "start_server"]
