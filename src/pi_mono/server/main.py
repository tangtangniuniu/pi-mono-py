"""Uvicorn server entry point — the pi-server CLI command."""

from __future__ import annotations

import click
import uvicorn


@click.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8080, type=int, help="Bind port")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload")
def main(host: str, port: int, reload: bool) -> None:
    """Start the Pi Agent REST server."""
    uvicorn.run(
        "pi_mono.server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
