"""Create, monitor, and use an OpenAI supervised fine-tuning job.

This example targets the real OpenAI API. Local emullm validates and records
jobs but returns a terminal training_not_available failure without training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")


def show_job(job: Any) -> None:
    error = None
    if job.error:
        error = {
            "code": job.error.code,
            "message": job.error.message,
            "param": job.error.param,
        }
    print(
        json.dumps(
            {
                "id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
                "error": error,
            },
            indent=2,
        )
    )


def upload(client: Any, path: Path) -> str:
    if path.suffix.lower() != ".jsonl":
        raise SystemExit(f"Training files must use the .jsonl extension: {path}")
    if not path.is_file():
        raise SystemExit(f"No such training file: {path}")
    with path.open("rb") as file_handle:
        uploaded = client.files.create(file=file_handle, purpose="fine-tune")
    print(f"Uploaded {path} as {uploaded.id}", file=sys.stderr)
    return uploaded.id


def create_job(args: argparse.Namespace, client: Any) -> int:
    training_file_id = upload(client, args.training_file)
    parameters: dict[str, Any] = {
        "model": args.model,
        "training_file": training_file_id,
        "method": {"type": "supervised"},
    }
    if args.validation_file:
        parameters["validation_file"] = upload(client, args.validation_file)

    job = client.fine_tuning.jobs.create(**parameters)
    show_job(job)
    print(f"Monitor with: python {Path(__file__).name} wait {job.id}", file=sys.stderr)
    return 0


def status_job(args: argparse.Namespace, client: Any) -> int:
    show_job(client.fine_tuning.jobs.retrieve(args.job_id))
    return 0


def wait_for_job(args: argparse.Namespace, client: Any) -> int:
    while True:
        job = client.fine_tuning.jobs.retrieve(args.job_id)
        print(f"{job.id}: {job.status}", file=sys.stderr)
        if job.status in TERMINAL_STATES:
            show_job(job)
            return 0 if job.status == "succeeded" else 1
        time.sleep(args.interval)


def cancel_job(args: argparse.Namespace, client: Any) -> int:
    show_job(client.fine_tuning.jobs.cancel(args.job_id))
    return 0


def chat(args: argparse.Namespace, client: Any) -> int:
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": "Classify support requests. Return exactly one lowercase label.",
            },
            {"role": "user", "content": args.prompt},
        ],
    )
    print(response.choices[0].message.content)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="upload JSONL data and create a supervised job")
    create.add_argument("--model", required=True, help="a currently supported fine-tunable base model")
    create.add_argument("--training-file", type=Path, required=True)
    create.add_argument("--validation-file", type=Path)
    create.set_defaults(handler=create_job)

    status = commands.add_parser("status", help="retrieve one job")
    status.add_argument("job_id")
    status.set_defaults(handler=status_job)

    wait = commands.add_parser("wait", help="poll until a job reaches a terminal state")
    wait.add_argument("job_id")
    wait.add_argument("--interval", type=float, default=30.0)
    wait.set_defaults(handler=wait_for_job)

    cancel = commands.add_parser("cancel", help="cancel a job")
    cancel.add_argument("job_id")
    cancel.set_defaults(handler=cancel_job)

    use = commands.add_parser("chat", help="send a request to a completed fine-tuned model")
    use.add_argument("--model", required=True, help="the ft:... model ID returned by a successful job")
    use.add_argument("prompt")
    use.set_defaults(handler=chat)
    return root


def main() -> int:
    args = parser().parse_args()
    require_api_key()
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the SDK with: python -m pip install --upgrade openai") from exc
    return args.handler(args, OpenAI())


if __name__ == "__main__":
    raise SystemExit(main())
