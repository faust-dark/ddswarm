"""Manual verification: streamable-HTTP transport end-to-end (real process + real HTTP)."""

import socket
import subprocess
import sys
import time

import anyio
from mcp.client.client import Client

PORT = 8765
BIN = "/home/faust/ddswarm/.venv/bin/ddswarm-mcp"


def wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


proc = subprocess.Popen(
    [BIN, "--transport", "streamable-http", "--port", str(PORT)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

try:
    if not wait_port(PORT):
        print("SERVER DID NOT COME UP")
        print(proc.stderr.read().decode())
        sys.exit(1)

    async def run() -> None:
        async with Client(f"http://127.0.0.1:{PORT}/mcp") as client:
            tools = await client.list_tools()
            print(f"{len(tools.tools)} tools over streamable HTTP")
            res = await client.call_tool("receptor_lookup", {"name": "GABRA1"})
            sc = getattr(res, "structured_content", None)
            print("receptor_lookup ->", sc["artifacts"]["receptor"]["gene"] if isinstance(sc, dict) else sc)
            res2 = await client.call_tool("screen_molecules", {"smiles": ["c1ccccc1", "CC(C"]})
            sc2 = getattr(res2, "structured_content", None)
            if isinstance(sc2, dict):
                s = sc2["artifacts"]["screened_batch"]["summary"]
                print(f"screen_molecules -> total={s['total']} passed={s['passed']} invalid={s['invalid']}")

    anyio.run(run)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
