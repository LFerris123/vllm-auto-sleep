# vLLM Auto-Sleep Proxy

A small asynchronous reverse proxy that lets a vLLM server release most GPU
memory while idle and wake automatically on the next inference request.

```text
client  ->  proxy :8001  ->  vLLM :8000
```

The proxy uses vLLM Sleep Mode Level 1. It forwards ordinary API traffic,
streams SSE responses without buffering, hides vLLM management endpoints, and
coordinates concurrent requests so that only one wake-up operation runs at a
time.

## Features

- Sleeps vLLM after a configurable period without inference traffic.
- Wakes vLLM before forwarding the next request.
- Streams response bodies, including server-sent events, chunk by chunk.
- Counts only inference endpoints as activity; health and model-list probes do
  not keep the model awake.
- Treats OpenAI-compatible endpoints and `/v1/messages` as inference traffic.
- Returns `503` when the upstream is unavailable instead of crashing.
- Returns `404` for sleep, wake, profiling, cache mutation, LoRA mutation, and
  other management endpoints.
- Binds to loopback by default and ignores ambient proxy environment variables
  for upstream traffic.

## Requirements

- Python 3.10 or newer
- vLLM with Sleep Mode support
- A vLLM process started with:
  - `VLLM_SERVER_DEV_MODE=1`
  - `--enable-sleep-mode`
- Enough host memory for Level 1 sleep to offload the model weights

The vLLM management API should remain bound to a trusted loopback interface.
Expose only the proxy endpoint through your chosen application gateway.

## Quick start

Start vLLM first:

```bash
export VLLM_SERVER_DEV_MODE=1
vllm serve /path/to/model \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-sleep-mode

# Keep the rest of your existing vLLM arguments in the command above.
```

Create an isolated Python environment and start the proxy:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python proxy.py
```

Point clients that previously used `http://127.0.0.1:8000` to:

```text
http://127.0.0.1:8001
```

Check proxy health:

```bash
curl --fail --silent --show-error http://127.0.0.1:8001/proxy/health
```

## Configuration

All configuration is provided through environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_BASE` | `http://127.0.0.1:8000` | Internal vLLM base URL |
| `PROXY_HOST` | `127.0.0.1` | Proxy listen address |
| `PROXY_PORT` | `8001` | Proxy listen port |
| `IDLE_SLEEP_AFTER` | `300` | Idle seconds before Level 1 sleep |
| `WAKE_TIMEOUT` | `120` | Maximum sleep/wake transition time |
| `SLEEP_LEVEL` | `1` | Sleep level; this proxy intentionally accepts only Level 1 |
| `LOG_LEVEL` | `INFO` | Python log level |

Example:

```bash
IDLE_SLEEP_AFTER=600 PROXY_PORT=8001 python proxy.py
```

## Activity rules

Only `POST` requests to these paths refresh the idle timer:

- `/v1/chat/completions`
- `/v1/chat/completions/batch`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/responses`
- `/v1/messages`

Model discovery, health checks, token counting, and monitoring do not refresh
the timer. Active inference streams prevent sleep until their cleanup finishes.

## Tests

The test suite uses a mock upstream and requires no GPU or model download:

```bash
python -m pip install -r requirements.txt
python tests/sleep_proxy_mock_e2e.py
```

It covers idle sleep, lazy wake, concurrent wake deduplication, wake timeout,
SSE forwarding, downstream disconnect cleanup, inference accounting, encoded
management-path blocking, and resistance to inherited proxy variables.

## Operational notes

### Why is the first request after an idle period slower?

The proxy waits for vLLM to restore model weights to GPU memory before it
forwards the request. Concurrent requests share the same wake-up task.

### How do I confirm that sleep released GPU memory?

After the idle threshold expires, query vLLM directly on its loopback listener:

```bash
curl --silent http://127.0.0.1:8000/is_sleeping
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

The sleep response should report `true`, and GPU memory use should fall
substantially. Exact savings depend on the model and engine configuration.

### Can I use Level 2 sleep?

No. This project deliberately uses Level 1 because it supports transparent
wake-up without reloading weights from their original storage. If host memory
cannot hold the model weights, choose a different lifecycle strategy instead
of changing `SLEEP_LEVEL`.

## Security boundary

The proxy blocks known vLLM management and development endpoints, including
encoded-path variants. It is still an application component, not a complete
network security product. Keep vLLM on loopback, place authentication and
transport security at the application gateway, and do not expose the raw vLLM
management listener.
