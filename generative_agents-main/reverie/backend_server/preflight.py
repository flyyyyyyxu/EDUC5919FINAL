"""
Lightweight preflight checks for the Generative Agents local setup.

Run this before starting the backend simulation server so configuration
problems fail fast with actionable messages.
"""
from pathlib import Path
import hashlib
import sys


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"


def _read_env(env_path):
  values = {}
  if not env_path.exists():
    return values

  for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    values[key] = value
  return values


def _mask_value(value, visible=6):
  if not value:
    return "<missing>"
  if len(value) <= visible * 2:
    return value[:visible] + "..."
  return value[:visible] + "..." + value[-visible:]


def _digest_value(value):
  if not value:
    return "<missing>"
  return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _check_imports():
  required = ("numpy", "requests", "django", "selenium")
  missing = []
  for mod in required:
    try:
      __import__(mod)
    except Exception:
      missing.append(mod)
  return missing


def main():
  problems = []

  missing_imports = _check_imports()
  if missing_imports:
    problems.append(
      "Missing Python packages in the active interpreter: "
      + ", ".join(missing_imports)
    )

  env_values = _read_env(ENV_PATH)
  if not ENV_PATH.exists():
    problems.append(f"Missing .env file at {ENV_PATH}")

  key = (
    env_values.get("MINIMAX_API_KEY")
    or env_values.get("MINIMAX_KEY")
    or env_values.get("OPENAI_API_KEY")
  )
  if not key:
    problems.append("No API key found in .env.")
  elif "replace_with_your_real" in key.lower():
    problems.append("API key in .env is still the placeholder value.")

  base = env_values.get("MINIMAX_API_BASE", "https://api.minimax.io/v1")
  model = env_values.get("MINIMAX_MODEL", env_values.get("MINIMAX_TEXT_MODEL", ""))
  if not base:
    problems.append("MINIMAX_API_BASE is empty.")
  if not model:
    problems.append("No MiniMax model configured in .env.")

  if problems:
    print("Preflight FAILED")
    for problem in problems:
      print(f"- {problem}")
    return 1

  print("Preflight OK")
  print(f"- env: {ENV_PATH}")
  print(f"- base: {base}")
  print(f"- model: {model}")
  print(f"- key: {_mask_value(key)}")
  print(f"- key_sha12: {_digest_value(key)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
