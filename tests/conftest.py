import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Tests run offline. A test that needs the internet must be marked `network`."""
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
