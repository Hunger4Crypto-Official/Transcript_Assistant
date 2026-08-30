"""
What the icon runs.

Double-clicking the app lands here: start the loopback server, open the person's
browser at it, and stay alive until they close the window that this prints how to
close. There is deliberately almost nothing here -- the server and the controller
hold the behaviour, so this stays a launcher and not a second place where things
can go subtly wrong.

Phone mode (`--phone`, or PLAUD_BRIDGE_PHONE=1 for a shortcut that cannot pass
flags) additionally binds the machine's Wi-Fi address, so a phone on the same
network can open the app and "Add to Home Screen" it. It is opt-in every single
launch: a tool holding recordings does not answer the network because it
answered it yesterday.
"""

from __future__ import annotations

import os
import socket
import sys
import webbrowser

from .controller import AppController
from .server import AppServer


def _lan_ip() -> str:
    """
    The address a phone on this network would reach this machine at.

    The connect() never sends a packet -- UDP connect only picks the local
    interface a route would use, which is exactly the answer needed. Falls back
    to hostname resolution, and to loopback when the machine has no network at
    all (phone mode then degrades to a printed explanation, not a crash).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 1))     # TEST-NET; never actually sent
            return probe.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def build(base_dir=None, host: str = "127.0.0.1", port: int = 0, phone: bool = False):
    """
    Stand up the server without opening a browser or blocking.

    Split out from `main` so a test can get the URL and the server object without
    launching a browser or serving forever. With `phone=True` the server binds
    every interface and `app.lan_url` carries the address a phone opens; without
    it the bind stays loopback-only, exactly as before.
    """
    app = AppServer(AppController(base_dir=base_dir))
    bind_host = "0.0.0.0" if phone else host
    httpd = app.make_server(bind_host, port)
    bound_port = httpd.server_address[1]
    url = f"http://127.0.0.1:{bound_port}/?token={app.token}"
    if phone:
        lan = _lan_ip()
        if lan != "127.0.0.1":
            app.enable_phone(lan, bound_port)
    return app, httpd, url


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    no_browser = "--no-browser" in argv
    phone = "--phone" in argv or os.environ.get("PLAUD_BRIDGE_PHONE", "").strip() not in ("", "0")

    app, httpd, url = build(phone=phone)
    print("Plaud Bridge is running.")
    print(f"  Open this in your browser if it did not open by itself:\n    {url}")
    if app.lan_url:
        print(f"  On your phone (same Wi-Fi):\n    {app.lan_url}")
        print("  Home network only -- the link carries this session's key.")
    elif phone:
        print("  Phone mode was requested but no network address was found; "
              "the app is loopback-only this launch.")
    print("  Keep this window open while you use it. Close it to stop.")

    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a headless box just prints the URL instead
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
