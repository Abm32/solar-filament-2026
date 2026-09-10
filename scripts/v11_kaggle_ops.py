"""Private V11 launches with source checks and a durable one-attempt receipt.

Run using the Python environment that contains kaggle. The installed SDK ignores
version suffixes for status/log/output, so every observation checks latest source
and version before and after the operation. Never push another version while
observing a run. An uncertain push leaves its receipt in place and is not retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def quota_snapshot(response):
    # kagglesdk.to_json() drops timedelta.days; use total_seconds throughout.
    result = {"refresh_utc": response.quota_refresh_time.isoformat()}
    for name in ("gpu_quota", "tpu_quota"):
        quota = getattr(response, name)
        values = {field: getattr(quota, field).total_seconds() for field in
                  ("time_used", "time_reserved", "total_time_allowed")}
        values["remaining_seconds"] = (values["total_time_allowed"]
                                       - values["time_used"] - values["time_reserved"])
        result[name] = values
    return result


def remote_identity(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug = ref.split("/")
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    return {"ref": response.metadata.ref,
            "version": response.metadata.current_version_number,
            "private": response.metadata.is_private,
            "sha256": hashlib.sha256(response.blob.source.encode()).hexdigest()}


def normalize_ref(value):
    """Push responses return a site-relative '/code/<owner>/<slug>' path.

    Every other endpoint returns the bare '<owner>/<slug>' ref, so compare on a
    single normalized form instead of failing an otherwise successful push.
    """
    text = str(value or "").strip()
    for prefix in ("https://www.kaggle.com/", "http://www.kaggle.com/", "/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.startswith("code/"):
        text = text[len("code/"):]
    return text.strip("/")


def own_refs(api):
    refs, token = set(), None
    while True:
        response = api.kernels_list_with_response(mine=True, page_size=100, page_token=token)
        refs.update(kernel.ref for kernel in response.kernels)
        token = response.next_page_token
        if not token:
            return refs


def verify_identity(identity, ref, digest, receipt):
    if identity["ref"] != ref or identity["sha256"] != digest or not identity["private"]:
        raise ValueError("Remote identity differs from the private staged source; use a new slug.")
    if receipt.get("version") is not None and identity["version"] != receipt["version"]:
        raise ValueError("Remote version changed; latest-session observations would be ambiguous.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "launch", "status", "logs", "output"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--file-pattern", default=r"\.(csv|json|log|pt)$")
    args = parser.parse_args()
    metadata = json.loads((args.stage / "kernel-metadata.json").read_text())
    ref = metadata["id"]
    digest = hashlib.sha256((args.stage / metadata["code_file"]).read_bytes()).hexdigest()
    receipt_path = args.stage / "kaggle_launch_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if str(metadata.get("is_private")).lower() != "true":
        raise ValueError("Only private staged notebooks are permitted.")
    if not ref.startswith(("abhimanyu122/solar-filament-v11-", "abhimanyu122/solar-filament-v12-",
                           "abhimanyu122/solar-filament-v13-", "abhimanyu122/solar-filament-v14-",
                           "abhimanyu122/solar-filament-v15-", "abhimanyu122/solar-filament-v16-",
                           "abhimanyu122/solar-filament-v17-", "abhimanyu122/solar-filament-v18-",
                           "abhimanyu122/solar-filament-v19-", "abhimanyu122/solar-filament-ea-")):
        raise ValueError("Expected a dedicated abhimanyu122 V11/V12/V13/V14/V15/V16/V17/V18/V19/EA notebook slug.")
    if receipt and (receipt["ref"] != ref or receipt["sha256"] != digest):
        raise ValueError("Staged source changed after a launch attempt; use a new stage and slug.")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    quota = quota_snapshot(api.quota_view()) if args.action in ("preflight", "launch") else None
    if args.action == "preflight":
        exists = ref in own_refs(api)
        identity = remote_identity(api, ref) if exists else None
        print(json.dumps({"ref": ref, "sha256": digest, "quota": quota,
                          "existing_remote": identity, "receipt": receipt}, indent=2))
        return
    if args.action == "launch":
        if ref in own_refs(api):
            identity = remote_identity(api, ref)
            verify_identity(identity, ref, digest, receipt)
            receipt.update(ref=ref, sha256=digest, version=identity["version"],
                           state="existing_run_verified")
            receipt_path.write_text(json.dumps(receipt, indent=2))
            print(json.dumps({"reused_existing_run": True, **identity}))
            return
        cap = 900 if "smoke" in ref else 10800
        if args.timeout is None or not 0 < args.timeout <= cap:
            raise ValueError(f"An explicit timeout of 1..{cap} seconds is required.")
        accelerator = "gpu_quota" if str(metadata.get("enable_gpu")).lower() == "true" else None
        if accelerator and args.timeout > quota[accelerator]["remaining_seconds"] - 60:
            raise ValueError("Requested timeout exceeds remaining GPU quota plus a 60s reserve.")
        receipt = {"ref": ref, "sha256": digest, "timeout_seconds": args.timeout,
                   "attempt_utc": datetime.now(timezone.utc).isoformat(), "state": "attempt_started"}
        # Exclusive creation blocks concurrent/uncertain retries before any remote mutation.
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        response = api.kernels_push(str(args.stage), timeout=str(args.timeout))
        receipt.update(returned_ref=response.ref, version=response.version_number,
                       state="server_returned", error=response.error)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        if response.error or normalize_ref(response.ref) != ref or not response.version_number:
            raise ValueError("Push returned an unexpected result; inspect the receipt before proceeding.")
        print(json.dumps(receipt, indent=2))
        return
    if not receipt.get("version"):
        raise ValueError("A verified launch receipt is required; launch can adopt an identical existing run.")
    identity = remote_identity(api, ref)
    verify_identity(identity, ref, digest, receipt)
    status = api.kernels_status(ref).status.name
    if args.action == "status":
        result = {**identity, "status": status}
    elif args.action == "logs":
        result = api.kernels_logs(ref)
    else:
        if status != "COMPLETE" or args.output is None:
            raise ValueError("Output download requires COMPLETE status and --output.")
        files, _ = api.kernels_output(ref, str(args.output), file_pattern=args.file_pattern,
                                      page_size=200, quiet=True)
        result = {**identity, "downloaded_files": len(files), "output": str(args.output)}
    after = remote_identity(api, ref)
    if after != identity:
        raise ValueError("Remote identity changed during observation; discard this result.")
    print(result if isinstance(result, str) else json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except Exception as error:
        # HTTP exception strings may contain signed output-download URLs.
        raise SystemExit(f"{type(error).__name__}; inspect receipt/status before any retry.") from None
