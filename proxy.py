#!/usr/bin/env python3
"""Transparent vLLM proxy with idle Level-1 sleep and lazy wake-up."""
from __future__ import annotations
import asyncio, logging, os, posixpath, time
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Awaitable, Callable, Iterable
from urllib.parse import unquote
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
def _positive(name: str, default: str, kind):
    value = kind(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
VLLM_BASE = os.getenv("VLLM_BASE", "http://127.0.0.1:8000").rstrip("/")
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = _positive("PROXY_PORT", "8001", int)
IDLE_SLEEP_AFTER = _positive("IDLE_SLEEP_AFTER", "300", float)
WAKE_TIMEOUT = _positive("WAKE_TIMEOUT", "120", float)
SLEEP_LEVEL = _positive("SLEEP_LEVEL", "1", int)
if SLEEP_LEVEL != 1: raise ValueError("SLEEP_LEVEL must be 1 for transparent wake-up")
CHECK_INTERVAL, POLL_INTERVAL, MANAGEMENT_TIMEOUT = 10.0, 0.5, min(10.0, WAKE_TIMEOUT)
# Exact matching keeps /v1/messages/count_tokens from becoming inference.
INFERENCE_PATHS = set(
    "/v1/chat/completions /v1/chat/completions/batch /v1/completions "
    "/v1/embeddings /v1/responses /v1/messages".split()
)
# VLLM_SERVER_DEV_MODE exposes much more than sleep APIs. Hide all core dev,
# mutation, profiling, LoRA, topology, and documentation endpoints.
BLOCKED_PATHS = set(
    """
    /reset_prefix_cache /reset_mm_cache /reset_encoder_cache
    /pause /resume /abort_requests /is_paused /init_weight_transfer_engine
    /start_weight_update /start_draft_weight_update /update_weights
    /finish_weight_update /update_weight_version /weight_info /get_world_size
    /collective_rpc /server_info /sleep /wake_up /is_sleeping
    /start_profile /stop_profile /v1/load_lora_adapter /v1/unload_lora_adapter
    /scale_elastic_ep /is_scaling_elastic_ep /fault_tolerance/apply
    /fault_tolerance/status /docs /docs/oauth2-redirect /redoc /openapi.json
    """.split()
)
HOP_BY_HOP = {
    name.encode()
    for name in "connection keep-alive proxy-authenticate proxy-authorization te "
    "proxy-connection trailer transfer-encoding upgrade host content-length".split()
}
logger = logging.getLogger("vllm-sleep-proxy")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
client: httpx.AsyncClient | None = None; idle_task: asyncio.Task[None] | None = None; wake_task: asyncio.Task[None] | None = None
transition_lock = asyncio.Lock()
last_inference_activity = time.monotonic()
active_requests = active_inference_requests = 0
class UpstreamUnavailable(RuntimeError): pass
def _normalize_path(path: str) -> str:
    # Canonicalize duplicate slashes and dot segments before denylist checks.
    return posixpath.normpath("/" + path.lstrip("/"))
def _path_variants(request: Request) -> Iterable[str]:
    raw = request.scope.get("raw_path", b"")
    value = raw.decode("latin1").split("?", 1)[0] if raw else request.url.path
    seen: set[str] = set()
    for _ in range(8):
        if value in seen:
            break
        seen.add(value)
        yield _normalize_path(value)
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    else:
        raise ValueError("path encoding exceeds safety limit")
    yield _normalize_path(request.url.path)
def _is_blocked(path: str) -> bool:
    path = _normalize_path(path)
    return (
        path in BLOCKED_PATHS
        or path == "/proxy"
        or path.startswith(("/metrics", "/proxy/", "/static", "/fault_tolerance/"))
    )
def _filtered_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    headers = list(raw_headers)
    connection_tokens: set[bytes] = set()
    for name, value in headers:
        if name.lower() == b"connection":
            connection_tokens.update(
                token.strip().lower() for token in value.split(b",") if token.strip()
            )
    blocked = HOP_BY_HOP | connection_tokens
    return [(name, value) for name, value in headers if name.lower() not in blocked]
def _http_client() -> httpx.AsyncClient:
    if client is None:
        raise RuntimeError("proxy HTTP client is not initialized")
    return client
async def _query_sleeping() -> bool:
    try:
        response = await _http_client().get(
            f"{VLLM_BASE}/is_sleeping", timeout=MANAGEMENT_TIMEOUT
        )
        response.raise_for_status()
        value = response.json()["is_sleeping"]
        if not isinstance(value, bool):
            raise ValueError("invalid is_sleeping response")
        return value
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise UpstreamUnavailable("vLLM sleep endpoint unavailable") from exc
async def _wake_sequence() -> None:
    if not await _query_sleeping():
        return
    started = time.monotonic()
    response = await _http_client().post(f"{VLLM_BASE}/wake_up")
    response.raise_for_status()
    while await _query_sleeping():
        await asyncio.sleep(POLL_INTERVAL)
    logger.info("vLLM wake completed in %.2fs", time.monotonic() - started)
async def _wake_with_deadline() -> None:
    try:
        await asyncio.wait_for(_wake_sequence(), timeout=WAKE_TIMEOUT)
    except asyncio.TimeoutError as exc:
        logger.error("vLLM wake timed out after %.1fs", WAKE_TIMEOUT)
        raise UpstreamUnavailable("vLLM wake timed out") from exc
    except httpx.HTTPError as exc:
        logger.error("vLLM wake request failed")
        raise UpstreamUnavailable("vLLM wake failed") from exc
async def _begin_request(is_inference: bool) -> None:
    global active_requests, active_inference_requests, last_inference_activity, wake_task
    async with transition_lock:
        active_requests += 1
        if is_inference:
            active_inference_requests += 1
            last_inference_activity = time.monotonic()
        if wake_task is None:
            wake_task = asyncio.create_task(_wake_with_deadline(), name="vllm-wake")
        task = wake_task
    try:
        await asyncio.shield(task)
    except BaseException:
        await _end_request(is_inference)
        raise
    finally:
        if task.done():
            async with transition_lock:
                if wake_task is task:
                    wake_task = None
async def _end_request(is_inference: bool) -> None:
    global active_requests, active_inference_requests, last_inference_activity
    async with transition_lock:
        active_requests = max(0, active_requests - 1)
        if is_inference:
            active_inference_requests = max(0, active_inference_requests - 1)
            last_inference_activity = time.monotonic()
async def _finish_request(upstream: httpx.Response | None, is_inference: bool) -> None:
    try:
        if upstream is not None:
            await upstream.aclose()
    except Exception:
        logger.warning("failed to close upstream response")
    finally:
        await _end_request(is_inference)
async def _maybe_sleep() -> None:
    async with transition_lock:
        idle_for = time.monotonic() - last_inference_activity
        waking = wake_task is not None and not wake_task.done()
        if active_inference_requests or waking or idle_for <= IDLE_SLEEP_AFTER:
            return
        try:
            if await _query_sleeping():
                return
            response = await asyncio.wait_for(
                _http_client().post(
                    f"{VLLM_BASE}/sleep", params={"level": SLEEP_LEVEL}
                ),
                timeout=WAKE_TIMEOUT,
            )
            response.raise_for_status()
            logger.info(
                "vLLM entered sleep level %d after %.1fs idle",
                SLEEP_LEVEL,
                idle_for,
            )
        except (asyncio.TimeoutError, httpx.HTTPError, UpstreamUnavailable):
            logger.warning("vLLM sleep attempt failed")
async def _idle_loop() -> None:
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        try:
            await _maybe_sleep()
        except Exception:
            logger.exception("unexpected idle-loop failure")
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global client, idle_task, wake_task, last_inference_activity
    timeout = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=10.0)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
    last_inference_activity = time.monotonic()
    idle_task = asyncio.create_task(_idle_loop(), name="vllm-idle-sleep")
    try:
        yield
    finally:
        idle_task.cancel()
        with suppress(asyncio.CancelledError):
            await idle_task
        if wake_task is not None:
            wake_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await wake_task
            wake_task = None
        await client.aclose()
        idle_task = None
        client = None
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
class ManagedStreamingResponse(StreamingResponse):
    def __init__(self, *args, cleanup: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup, self.cleaned = cleanup, False
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self.cleaned:
                self.cleaned = True
                task = asyncio.create_task(self.cleanup())
                cancelled = False
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        cancelled = True
                task.result()
                if cancelled:
                    raise asyncio.CancelledError
@app.get("/proxy/health")
async def proxy_health() -> JSONResponse:
    async with transition_lock:
        idle_for = max(0.0, time.monotonic() - last_inference_activity)
        payload = {
            "seconds_since_last_inference": round(idle_for, 3),
            "active_requests": active_requests,
            "active_inference_requests": active_inference_requests,
        }
        try:
            sleeping = await _query_sleeping()
        except UpstreamUnavailable:
            payload.update(status="degraded", upstream="unavailable", upstream_is_sleeping=None)
            return JSONResponse(status_code=503, content=payload)
        payload.update(status="ok", upstream="ok", upstream_is_sleeping=sleeping)
        return JSONResponse(content=payload)
@app.api_route("/", methods=["GET", "POST"])
@app.api_route("/{path:path}", methods=["GET", "POST"])
async def forward(request: Request, path: str = ""):
    request_path = request.url.path
    try:
        path_variants = set(_path_variants(request))
    except ValueError:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    normalized_path = _normalize_path(request_path)
    if any(_is_blocked(candidate) for candidate in path_variants):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    is_inference = request.method == "POST" and bool(path_variants & INFERENCE_PATHS)
    try:
        await _begin_request(is_inference)
    except UpstreamUnavailable:
        logger.warning("upstream unavailable before %s %s", request.method, normalized_path)
        return JSONResponse(status_code=503, content={"detail": "vLLM unavailable"})
    upstream: httpx.Response | None = None
    try:
        url = httpx.URL(f"{VLLM_BASE}{request_path}").copy_with(
            query=request.scope.get("query_string", b"")
        )
        upstream_request = _http_client().build_request(
            request.method,
            url,
            headers=_filtered_headers(request.headers.raw),
            content=await request.body(),
        )
        upstream = await _http_client().send(upstream_request, stream=True)
    except asyncio.CancelledError:
        await _finish_request(upstream, is_inference)
        raise
    except Exception:
        await _finish_request(upstream, is_inference)
        logger.warning("upstream connection failed for %s %s", request.method, normalized_path)
        return JSONResponse(status_code=503, content={"detail": "vLLM unavailable"})
    assert upstream is not None
    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            logger.warning("upstream stream failed for %s %s", request.method, normalized_path)
    response = ManagedStreamingResponse(
        relay(), status_code=upstream.status_code,
        cleanup=lambda: _finish_request(upstream, is_inference),
    )
    response.raw_headers = _filtered_headers(upstream.headers.raw)
    return response
if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, access_log=False)
