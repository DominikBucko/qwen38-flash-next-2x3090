# Image inputs

The checkpoint includes the BF16 vision encoder. No extra model download is
needed. Image inputs are opt-in; text-only serving stays the default.

## Enable images

Set these values in `.env` (also listed in [`configs/vision.env`](../configs/vision.env)):

```dotenv
ENABLE_VISION=1
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=80
VISION_MAX_IMAGES=1
VISION_MAX_PIXELS=1048576
```

Keep `MODEL_DIR` and the usual single-request 256K settings. Rebuild and restart
with `make serve`, or use `docker compose -f docker/compose.yaml up --build`.
Export the same values when calling `scripts/docker_serve.sh` directly.

Hot80 leaves room for the vision encoder and its working memory. It does not
remove experts or change their precision; uncached experts still come from RAM.
The encoder runs on the GPUs with its weights split across both cards. This is
not CPU vision offload. MTP3, BF16 KV, and the 262,144-token limit stay enabled.

The profile permits one image per request and disables video. The processor
resizes images to at most 1,048,576 pixels, preserving aspect ratio subject to
patch alignment. This bounds encoder work, not the size of an uploaded file.
Larger limits need new memory checks. Image tokens, text, and output all count
toward the context limit.
The resolution cap can lose fine text in large screenshots; crop the relevant
region before uploading when detail matters.

## Send a local image

Use the OpenAI-compatible `image_url` content type. A data URL keeps the request
self-contained. This example needs only Python's standard library:

```bash
python3 - screenshot.png <<'PY'
import base64, json, mimetypes, pathlib, sys, urllib.request

path = pathlib.Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0] or "image/png"
url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
payload = {
    "model": "Qwen3.8-Flash-Next", "max_tokens": 512, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Describe what is visible in this image."}
    ]}]
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PY
```

An `At most 0 image(s)` error means the server is still in text-only mode.
Check the container's `ENABLE_VISION` value and rebuild after updating the
launcher. `At most 1 image(s)` means the request exceeds this profile's limit.

If image prefill runs out of GPU memory, first reduce `VISION_MAX_PIXELS` to
`262144`, or lower the hot cache further. Do not reduce model or KV precision as
an OOM workaround. See [memory tuning](memory.md) for host RAM and swap.

## What was tested

On two RTX 3090s and 128 GB RAM, the Docker profile read the printed codes and
left/right colors correctly in three synthetic images. Two were 768x512; the
third was 1536x1024 and needed downscaling. The one-image limit rejected a
two-image request. MTP draft and accepted-token counters advanced on the image
requests.

With the encoder still loaded, the server completed 262,016 text input + 128
output tokens, then two short text checks. Counts matched, with no inference
allocation retries or preemptions. The KV pool remained 276,313 tokens.

These are basic grounding and capacity checks. They do not cover full-context
mixed image/text prompts, concurrent vision users, video, CPU encoder offload,
or broad vision quality. No matched speed comparison was run. The
[validation record](../benchmarks/2026-09-16/vision.json) has the settings and results.
