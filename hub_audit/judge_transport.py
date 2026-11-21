"""Container-local TCP to host Unix socket transport; carries no model credentials."""

from __future__ import annotations

import contextlib
import select
import socket
import socketserver
import threading


@contextlib.contextmanager
def forward(socket_path: str):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                upstream.connect(socket_path)
                peers = {self.request: upstream, upstream: self.request}
                while True:
                    ready, _, _ = select.select(list(peers), [], [], 120)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        peers[source].sendall(data)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
