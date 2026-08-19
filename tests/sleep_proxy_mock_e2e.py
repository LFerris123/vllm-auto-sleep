#!/usr/bin/env python3
"""Isolated integration test for the sleep proxy; no GPU or vLLM required."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

mock_app = FastAPI()
mock_state = {"sleeping": False, "sleep_calls": 0, "wake_calls": 0, "hang_wake": False}


@mock_app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@mock_app.get("/is_sleeping")
async def is_sleeping():
    return {"is_sleeping": mock_state["sleeping"]}


@mock_app.post("/sleep")
async def sleep():
    mock_state["sleeping"] = True
    mock_state["sleep_calls"] += 1
    return JSONResponse(content=None)


@mock_app.post("/wake_up")
async def wake_up():
    mock_state["wake_calls"] += 1
    if mock_state["hang_wake"]:
        await asyncio.sleep(30)
    await asyncio.sleep(0.2)
    mock_state["sleeping"] = False
    return JSONResponse(content=None)


@mock_app.get("/mock/state")
async def state():
    return mock_state


@mock_app.post("/mock/sleep")
async def force_sleep():
    mock_state["sleeping"] = True
    return mock_state


@mock_app.post("/mock/hang_wake")
async def hang_wake():
    mock_state.update(sleeping=True, hang_wake=True)
    return mock_state


@mock_app.api_route("/{path:path}", methods=["GET", "POST"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    if request.headers.get("x-test-delayed-header") == "1":
        await asyncio.sleep(0.5)
    if request.headers.get("x-test-disconnect-stream") == "1":
        async def disconnect_chunks():
            for index in range(300):
                yield f"data: {index}\n\n".encode()
                await asyncio.sleep(0.1)

        return StreamingResponse(disconnect_chunks(), media_type="text/event-stream")
    if request.headers.get("x-test-stream") == "1":
        async def chunks():
            yield b"data: first\n\n"
            await asyncio.sleep(0.3)
            yield b"data: second\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")
    if request.headers.get("x-test-long-stream") == "1":
        async def long_chunks():
            yield b"data: open\n\n"
            await asyncio.sleep(11)
            yield b"data: close\n\n"

        return StreamingResponse(long_chunks(), media_type="text/event-stream")
    return {"path": f"/{path}", "query": request.url.query, "body": body.decode()}


def wait_http(url: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=0.5, trust_env=False).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise AssertionError(f"not ready: {url}")


def raw_status(path: str) -> int:
    with socket.create_connection(("127.0.0.1", 18081), timeout=2) as conn:
        request = f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        conn.sendall(request.encode())
        response = conn.recv(256).decode("latin1")
    return int(response.split(" ", 2)[1])


def disconnect_request(path: str, header: str, read_first_chunk: bool) -> None:
    with socket.create_connection(("127.0.0.1", 18081), timeout=2) as conn:
        request = (
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n"
            f"{header}: 1\r\nConnection: close\r\n\r\n"
        )
        conn.sendall(request.encode())
        if read_first_chunk:
            conn.recv(1024)


def raw_long_stream(path: str) -> int:
    with socket.create_connection(("127.0.0.1", 18081), timeout=20) as conn:
        request = (
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n"
            "x-test-long-stream: 1\r\nConnection: close\r\n\r\n"
        )
        conn.sendall(request.encode())
        response = b""
        while b"data: close" not in response:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response += chunk
    return int(response.split(b" ", 2)[1])


def wait_for_zero_counts(client: httpx.Client) -> None:
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        health = client.get("/proxy/health").json()
        if health["active_requests"] == health["active_inference_requests"] == 0:
            return
        time.sleep(0.1)
    raise AssertionError("request counters leaked after downstream disconnect")


def test_managed_cleanup(proxy_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location("proxy_under_test", proxy_dir / "proxy.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def one_case(fail_type: str) -> None:
        cleaned = 0

        async def body():
            yield b"chunk"

        async def cleanup():
            nonlocal cleaned
            cleaned += 1

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == fail_type:
                raise OSError("forced downstream disconnect")

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/",
            "raw_path": b"/", "query_string": b"", "headers": [],
            "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 2),
        }
        response = module.ManagedStreamingResponse(body(), cleanup=cleanup)
        try:
            await response(scope, receive, send)
        except Exception:
            pass
        assert cleaned == 1, f"cleanup count for {fail_type}: {cleaned}"

    asyncio.run(one_case("http.response.start"))
    asyncio.run(one_case("http.response.body"))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    proxy_dir = Path(os.environ.get("PROXY_SOURCE_DIR", repo))
    test_managed_cleanup(proxy_dir)
    test_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(
        VLLM_BASE="http://127.0.0.1:18080",
        PROXY_HOST="127.0.0.1",
        PROXY_PORT="18081",
        IDLE_SLEEP_AFTER="1",
        WAKE_TIMEOUT="1",
        SLEEP_LEVEL="1",
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        ALL_PROXY="http://127.0.0.1:9",
        NO_PROXY="",
    )
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    mock = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "sleep_proxy_mock_e2e:mock_app",
         "--app-dir", str(test_dir), "--host", "127.0.0.1", "--port", "18080"],
        **quiet,
    )
    proxy = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "proxy:app", "--app-dir", str(proxy_dir),
         "--host", "127.0.0.1", "--port", "18081", "--no-access-log"],
        env=env,
        **quiet,
    )
    try:
        wait_http("http://127.0.0.1:18080/health")
        wait_http("http://127.0.0.1:18081/proxy/health")
        client = httpx.Client(
            base_url="http://127.0.0.1:18081", timeout=20, trust_env=False
        )

        model = client.get("/v1/models?x=1")
        assert model.status_code == 200 and model.json()["query"] == "x=1"
        plain = client.post("/v1/chat/completions", content=b'{"hello":"world"}')
        assert plain.status_code == 200 and "hello" in plain.json()["body"]

        started = time.monotonic()
        with client.stream("POST", "/v1/chat/completions", headers={"x-test-stream": "1"}) as response:
            pieces = list(response.iter_raw())
        assert response.status_code == 200 and b"first" in b"".join(pieces)
        assert b"second" in b"".join(pieces) and time.monotonic() - started >= 0.25

        assert client.get("/sleep").status_code == 404
        assert client.get("/metrics").status_code == 404
        assert raw_status("/safe/../sleep") == 404
        assert raw_status("/%2573leep") == 404
        assert raw_status("/%252e%252e/sleep") == 404
        deep_path = "/sleep"
        for _ in range(9):
            deep_path = quote(deep_path, safe="")
        assert raw_status("/" + deep_path) == 404

        disconnect_request("/v1/messages", "x-test-disconnect-stream", True)
        wait_for_zero_counts(client)

        before = client.get("/proxy/health").json()["seconds_since_last_inference"]
        time.sleep(0.2)
        assert client.get("/v1/models").status_code == 200
        after = client.get("/proxy/health").json()["seconds_since_last_inference"]
        assert after > before

        client.post("/mock/sleep")
        wake_before = httpx.get(
            "http://127.0.0.1:18080/mock/state", trust_env=False
        ).json()["wake_calls"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            statuses = list(pool.map(lambda _: client.post("/v1/responses").status_code, range(3)))
        wake_after = httpx.get(
            "http://127.0.0.1:18080/mock/state", trust_env=False
        ).json()["wake_calls"]
        assert statuses == [200, 200, 200] and wake_after == wake_before + 1

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(raw_long_stream, "/v1/chat/%2563ompletions")
            time.sleep(10.3)
            assert not httpx.get(
                "http://127.0.0.1:18080/is_sleeping", trust_env=False
            ).json()["is_sleeping"]
            assert future.result() == 200

        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if httpx.get(
                "http://127.0.0.1:18080/is_sleeping", trust_env=False
            ).json()["is_sleeping"]:
                break
            time.sleep(0.2)
        else:
            raise AssertionError("idle loop did not sleep")

        httpx.post("http://127.0.0.1:18080/mock/hang_wake", trust_env=False)
        wake_before = httpx.get(
            "http://127.0.0.1:18080/mock/state", trust_env=False
        ).json()["wake_calls"]
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda _: client.post("/v1/responses").status_code, range(2)))
        wake_after = httpx.get(
            "http://127.0.0.1:18080/mock/state", trust_env=False
        ).json()["wake_calls"]
        assert statuses == [503, 503] and wake_after == wake_before + 1
        assert time.monotonic() - started < 1.8

        print(json.dumps({"result": "PASS", "wake_calls": wake_after}, sort_keys=True))
    finally:
        for process in (proxy, mock):
            process.terminate()
        for process in (proxy, mock):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
