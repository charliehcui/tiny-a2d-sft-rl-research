"""下载固定版本原始检查点，并验证上游哈希；不加载或修改模型。"""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "models" / "Qwen3-0.6B-diffusion-mdlm-v0.1"
MANIFEST = json.loads((PROJECT / "configs" / "baseline_model_files.json").read_text())


def valid_file(path, item):
    if not path.exists() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256() if item["sha256"] else hashlib.sha1()
    if not item["sha256"]:
        digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == (item["sha256"] or item["git_blob_oid"])


def download(item, endpoint):
    path = MODEL / item["path"]
    if valid_file(path, item):
        print(json.dumps({"file": item["path"], "status": "already_verified"}), flush=True)
        return
    partial = path.with_name(path.name + ".partial")
    url = f"{endpoint}/{MANIFEST['model_id']}/resolve/{MANIFEST['revision']}/{item['path']}?download=true"
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                last_report = time.monotonic()
                with partial.open("ab" if append else "wb") as output:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 20:
                            print(json.dumps({"file": item["path"], "downloaded_bytes": offset, "total_bytes": item["size"]}), flush=True)
                            last_report = time.monotonic()
            if not valid_file(partial, item):
                raise ValueError(f"Checksum mismatch: {item['path']}")
            partial.replace(path)
            print(json.dumps({"file": item["path"], "status": "downloaded_and_verified", "bytes": item["size"]}), flush=True)
            return
        except Exception as error:
            print(json.dumps({"file": item["path"], "attempt": attempt + 1, "error": str(error)}), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    endpoint = None
    for candidate in ["https://huggingface.co", "https://hf-mirror.com"]:
        url = f"{candidate}/{MANIFEST['model_id']}/resolve/{MANIFEST['revision']}/config.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                response.read()
            endpoint = candidate
            break
        except Exception as error:
            print(json.dumps({"endpoint": candidate, "probe_error": str(error)}), flush=True)
    if endpoint is None:
        raise RuntimeError("Neither model download endpoint is reachable")
    print(json.dumps({"endpoint": endpoint, "model_id": MANIFEST["model_id"], "revision": MANIFEST["revision"]}), flush=True)
    items = sorted(MANIFEST["files"], key=lambda item: -item["size"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download, item, endpoint) for item in items]
        for future in futures:
            future.result()
    MANIFEST["download_endpoint"] = endpoint
    (MODEL / "source_manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    (PROJECT / "work" / "model_ready").touch()
    print(json.dumps({"status": "model_ready"}), flush=True)


if __name__ == "__main__":
    main()
