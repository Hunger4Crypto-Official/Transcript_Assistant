"""
What the icon runs.

Double-clicking the app lands here: start the loopback server, open the person's
browser at it, and stay alive until they close the window that this prints how to
close. There is deliberately almost nothing here -- the server and the controller
hold the behaviour, so this stays a launcher and not a second place where things
can go subtly wrong.
"""

from __future__ import annotations

import sys
import webbrowser

from .controller import AppController
from .server import AppServer


def build(base_dir=None, host: str = "127.0.0.1", port: int = 0):
    """
    Stand up the server without opening a browser or blocking.

    Split out from `main` so a test can get the URL and the server object without
    launching a browser or serving forever.
    """
    app = AppServer(AppController(base_dir=base_dir))
    httpd = app.make_server(host, port)
    bound_host, bound_port = httpd.server_address
    url = f"http://{bound_host}:{bound_port}/?token={app.token}"
    return app, httpd, url


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    no_browser = "--no-browser" in argv

    _app, httpd, url = build()
    print("Plaud Bridge is running.")
    print(f"  Open this in your browser if it did not open by itself:\n    {url}")
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
