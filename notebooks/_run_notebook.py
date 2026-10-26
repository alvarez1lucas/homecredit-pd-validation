"""
Ejecuta una notebook de punta a punta y guarda los outputs en el mismo archivo.

Uso (desde la raíz del proyecto):
    python notebooks/_run_notebook.py notebooks/02_threshold_analysis.ipynb

Equivale a `jupyter nbconvert --execute --inplace`, pero sólo depende de
jupyter_client + ipykernel (ya instalados como dependencias de ipykernel).
"""
import json
import sys
import time
from pathlib import Path

from jupyter_client.manager import start_new_kernel


def run(path: Path, timeout: int = 7200) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
    ok, count = True, 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            count += 1
            msg_id = kc.execute(src)
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    if outputs and outputs[-1].get("output_type") == "stream" \
                            and outputs[-1]["name"] == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": content["name"],
                                        "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    out = {"output_type": kind, "data": content["data"],
                           "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = count
                    outputs.append(out)
                elif kind == "error":
                    outputs.append({"output_type": "error", "ename": content["ename"],
                                    "evalue": content["evalue"], "traceback": content["traceback"]})
                    ok = False
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            cell["outputs"] = outputs
            cell["execution_count"] = count
            if not ok:
                print(f"ERROR en celda {count}:\n{src[:500]}\n--> {outputs[-1]['ename']}: {outputs[-1]['evalue']}")
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return ok


if __name__ == "__main__":
    all_ok = True
    for p in sys.argv[1:]:
        t0 = time.time()
        ok = run(Path(p))
        all_ok &= ok
        print(f"{'OK ' if ok else 'FAIL'} {p} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if all_ok else 1)
