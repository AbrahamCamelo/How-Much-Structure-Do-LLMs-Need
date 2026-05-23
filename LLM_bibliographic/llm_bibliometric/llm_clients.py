from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
import requests
from openai import OpenAI

from .constants import DEBUG_DIR, DEFAULT_CHAT_MODELS, TOKEN_USAGE_CSV
from .utils import ensure_directory, safe_json_loads, slugify_filename, text_hash


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 8000
    timeout_seconds: int = 180


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        provider = config.provider.strip().lower()
        aliases = {
            "openai": "chatgpt",
            "chatgpt": "chatgpt",
            "google": "gemini",
            "gemini": "gemini",
            "anthropic": "claude",
            "claude": "claude",
        }
        if provider not in aliases:
            raise ValueError(f"Unsupported provider: {config.provider}")
        self.provider = aliases[provider]
        self.config = LLMConfig(
            provider=self.provider,
            model=config.model or DEFAULT_CHAT_MODELS[self.provider],
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout_seconds=config.timeout_seconds,
        )
        self._last_response_debug: dict[str, object] | None = None
        self._request_context: dict[str, str | None] = {
            "paper": None,
            "pipeline": None,
            "step": None,
        }

    def set_request_context(
        self,
        *,
        paper: str | None = None,
        pipeline: str | None = None,
        step: str | None = None,
    ) -> None:
        self._request_context = {
            "paper": paper,
            "pipeline": pipeline,
            "step": step,
        }

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self._last_response_debug = None
        if self.provider == "chatgpt":
            return self._generate_openai_text(system_prompt, user_prompt)
        if self.provider == "gemini":
            return self._generate_gemini_text(system_prompt, user_prompt)
        return self._generate_claude_text(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> object:
        response_text = self.generate_text(system_prompt, user_prompt)
        try:
            return safe_json_loads(response_text)
        except json.JSONDecodeError as error:
            debug_paths = self._save_json_debug_artifacts(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response_text,
                error=error,
            )
            if debug_paths:
                debug_listing = "\n".join(f"- {path}" for path in debug_paths)
                error.add_note(f"Saved JSON parse debug artifacts:\n{debug_listing}")
            raise

    def _debug_file_stem(self, user_prompt: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_slug = slugify_filename(self.config.model or "unknown_model")
        prompt_hash = text_hash(user_prompt)[:12]
        return f"{timestamp}__{self.provider}__{model_slug}__{prompt_hash}"

    def _save_json_debug_artifacts(
        self,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        error: json.JSONDecodeError,
    ) -> list[Path]:
        try:
            debug_dir = ensure_directory(DEBUG_DIR)
            stem = self._debug_file_stem(user_prompt)

            metadata_path = debug_dir / f"{stem}__metadata.json"
            prompt_path = debug_dir / f"{stem}__prompt.txt"
            response_path = debug_dir / f"{stem}__response.txt"
            provider_response_path = debug_dir / f"{stem}__provider_response.json"

            metadata = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "timeout_seconds": self.config.timeout_seconds,
                "user_prompt_hash": text_hash(user_prompt),
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "response_chars": len(response_text),
                "json_error": {
                    "message": error.msg,
                    "line": error.lineno,
                    "column": error.colno,
                    "position": error.pos,
                },
            }
            if self._last_response_debug is not None:
                metadata["provider_response_summary"] = self._summarize_provider_response_debug(
                    self._last_response_debug
                )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            prompt_path.write_text(
                (
                    "=== SYSTEM PROMPT ===\n"
                    f"{system_prompt}\n\n"
                    "=== USER PROMPT ===\n"
                    f"{user_prompt}\n"
                ),
                encoding="utf-8",
            )
            response_path.write_text(response_text, encoding="utf-8")
            written_paths = [metadata_path, prompt_path, response_path]
            if self._last_response_debug is not None:
                provider_response_path.write_text(
                    json.dumps(self._last_response_debug, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                written_paths.append(provider_response_path)
            return written_paths
        except Exception:
            return []

    def _summarize_provider_response_debug(
        self,
        provider_response_debug: dict[str, object],
    ) -> dict[str, object]:
        summary: dict[str, object] = {}
        prompt_feedback = provider_response_debug.get("prompt_feedback")
        if prompt_feedback is not None:
            summary["prompt_feedback"] = prompt_feedback

        candidates = provider_response_debug.get("candidates")
        if isinstance(candidates, list):
            summary["candidate_count"] = len(candidates)
            finish_reasons: list[object] = []
            finish_messages: list[object] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                finish_reasons.append(candidate.get("finish_reason"))
                if "finish_message" in candidate:
                    finish_messages.append(candidate.get("finish_message"))
            summary["finish_reasons"] = finish_reasons
            if finish_messages:
                summary["finish_messages"] = finish_messages
        return summary

    def _capture_gemini_response_debug(
        self,
        response: types.GenerateContentResponse,
    ) -> None:
        try:
            response_debug = response.model_dump(mode="json", exclude_none=False)
        except Exception:
            response_debug = {
                "prompt_feedback": str(getattr(response, "prompt_feedback", None)),
                "candidates": str(getattr(response, "candidates", None)),
            }
        self._last_response_debug = response_debug

    def _print_gemini_usage(
        self,
        response: types.GenerateContentResponse,
    ) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        print(
            (
                "[gemini usage] "
                f"model={self.config.model} "
                f"prompt_tokens={getattr(usage, 'prompt_token_count', None)} "
                f"output_tokens={getattr(usage, 'candidates_token_count', None)} "
                f"thinking_tokens={getattr(usage, 'thoughts_token_count', None)} "
                f"total_tokens={getattr(usage, 'total_token_count', None)}"
            )
        )

    def _append_token_usage_row(
        self,
        *,
        prompt_tokens: object,
        output_tokens: object,
        thinking_tokens: object,
        total_tokens: object,
    ) -> None:
        csv_path = TOKEN_USAGE_CSV
        ensure_directory(csv_path.parent)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "paper": self._request_context.get("paper") or "",
            "provider": self.provider,
            "model": self.config.model or "",
            "pipeline": self._request_context.get("pipeline") or "",
            "step": self._request_context.get("step") or "",
            "prompt_tokens": "" if prompt_tokens is None else prompt_tokens,
            "output_tokens": "" if output_tokens is None else output_tokens,
            "thinking_tokens": "" if thinking_tokens is None else thinking_tokens,
            "total_tokens": "" if total_tokens is None else total_tokens,
        }
        fieldnames = list(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _generate_openai_text(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        return response.output_text.strip()

    def _generate_gemini_text(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        with genai.Client(
            api_key=api_key,
        ) as client:
            response = client.models.generate_content(
                model=self.config.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.config.temperature,
                    response_mime_type="application/json",
                    max_output_tokens=self.config.max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=3000
                    ),
                ),
            )
        self._capture_gemini_response_debug(response)
        self._print_gemini_usage(response)
        usage = getattr(response, "usage_metadata", None)
        self._append_token_usage_row(
            prompt_tokens=getattr(usage, "prompt_token_count", None) if usage is not None else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage is not None else None,
            thinking_tokens=getattr(usage, "thoughts_token_count", None) if usage is not None else None,
            total_tokens=getattr(usage, "total_token_count", None) if usage is not None else None,
        )
        return response.text.strip()

    def _generate_claude_text(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.config.model,
                "system": system_prompt,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ).strip()
