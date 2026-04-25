"""
File: json_logger.py
Description: Structured JSON logging system for generative agent simulations.
Replaces all print()-based debug/verbose output with structured JSONL events.

Log files are written to: {fs_storage}/{sim_code}/logs/
  sim_run.jsonl    -- streaming event log (one JSON object per line)
  sim_summary.json -- written on simulation finish/save
"""
import json
import os
import datetime
import threading
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_logger: Optional["SimLogger"] = None


def init_logger(sim_code: str, fs_storage: str,
                log_level: str = "INFO") -> "SimLogger":
    """Create and register the global SimLogger. Call once from ReverieServer.__init__."""
    global _logger
    log_dir = os.path.join(fs_storage, sim_code, "logs")
    os.makedirs(log_dir, exist_ok=True)
    _logger = SimLogger(sim_code, log_dir, log_level.upper())
    return _logger


def get_logger() -> "SimLogger":
    """Return the active SimLogger. Returns a no-op logger if not yet initialised."""
    if _logger is None:
        return _NoOpLogger()
    return _logger


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
_LEVELS = {"DEBUG": 10, "INFO": 20, "ERROR": 40}


def _level_value(level: str) -> int:
    return _LEVELS.get(level.upper(), 20)


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------
class SimLogger:
    def __init__(self, sim_code: str, log_dir: str, log_level: str = "INFO"):
        self.sim_code = sim_code
        self.log_dir = log_dir
        self.log_level = log_level
        self._min_level = _level_value(log_level)
        self._jsonl_path = os.path.join(log_dir, "sim_run.jsonl")
        self._summary_path = os.path.join(log_dir, "sim_summary.json")
        self._lock = threading.Lock()
        self._current_step: Optional[int] = None
        self._start_wall: str = datetime.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Internal write
    # ------------------------------------------------------------------
    def _write(self, level: str, event: str,
               step: Optional[int], data: dict) -> None:
        if _level_value(level) < self._min_level:
            return
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "step": step if step is not None else self._current_step,
            "level": level,
            "event": event,
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------
    def set_step(self, step: int) -> None:
        self._current_step = step

    # ------------------------------------------------------------------
    # Public event methods
    # ------------------------------------------------------------------
    def log_step_complete(self, step: int, curr_time: str,
                          personas_state: dict) -> None:
        """INFO. Called after all persona.move() calls complete each step."""
        self._write("INFO", "step_complete", step, {
            "curr_time": curr_time,
            "personas": personas_state,
        })

    def log_conversation(self, init_persona: str, target_persona: str,
                         utterances: Any) -> None:
        """INFO. Called after a conversation is generated."""
        self._write("INFO", "conversation", None, {
            "init_persona": init_persona,
            "target_persona": target_persona,
            "utterances": utterances if isinstance(utterances, list) else str(utterances),
        })

    def log_daily_plan(self, persona_name: str, wake_up_hour: int,
                       daily_req: list) -> None:
        """INFO. Called after long-term daily planning."""
        self._write("INFO", "daily_plan", None, {
            "persona": persona_name,
            "wake_up_hour": wake_up_hour,
            "daily_req": daily_req,
        })

    def log_reflection(self, persona_name: str, insights: list) -> None:
        """INFO. Called after reflection generates new insights."""
        self._write("INFO", "reflection", None, {
            "persona": persona_name,
            "insights": insights,
        })

    def log_auto_save(self, step: int) -> None:
        """INFO. Replaces print(f'[Auto-saved at step {step}]')."""
        self._write("INFO", "auto_save", step, {"step": step})

    def log_gns_function(self, func_name: str,
                         persona_name: str = "") -> None:
        """DEBUG. Replaces all `if debug: print('GNS FUNCTION: <...>')` calls."""
        self._write("DEBUG", "gns_function", None, {
            "function": func_name,
            "persona": persona_name,
        })

    def log_prompt(self, template: str = "", persona_name: str = "",
                   gpt_param: dict = None, prompt_input: Any = None,
                   prompt: str = "", output: Any = None) -> None:
        """DEBUG. Replaces print_run_prompts() / verbose output."""
        self._write("DEBUG", "prompt_call", None, {
            "template": template,
            "persona": persona_name,
            "gpt_param": gpt_param or {},
            "prompt_input": str(prompt_input) if prompt_input is not None else None,
            "prompt": prompt,
            "output": str(output) if output is not None else None,
        })

    def log_error(self, context: str, error: str) -> None:
        """ERROR. For exceptions and API failures."""
        self._write("ERROR", "error", None, {
            "context": context,
            "error": error,
        })

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def finalize(self, total_steps: int, personas: list) -> None:
        """
        Read sim_run.jsonl, aggregate all INFO events, write sim_summary.json.
        Safe to call multiple times (auto-save checkpoints).
        """
        end_wall = datetime.datetime.utcnow().isoformat() + "Z"

        timeline = []
        conversations = []
        daily_plans: dict = {}
        reflections: dict = {}

        if not os.path.exists(self._jsonl_path):
            return

        with self._lock:
            with open(self._jsonl_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if rec.get("level") != "INFO":
                        continue

                    timeline.append(rec)
                    event = rec.get("event", "")
                    data = rec.get("data", {})

                    if event == "conversation":
                        conversations.append({
                            "step": rec.get("step"),
                            "participants": [
                                data.get("init_persona", ""),
                                data.get("target_persona", ""),
                            ],
                            "utterances": data.get("utterances", []),
                        })

                    elif event == "daily_plan":
                        persona = data.get("persona", "")
                        if persona not in daily_plans:
                            daily_plans[persona] = []
                        daily_plans[persona].append({
                            "step": rec.get("step"),
                            "wake_up_hour": data.get("wake_up_hour"),
                            "daily_req": data.get("daily_req", []),
                        })

                    elif event == "reflection":
                        persona = data.get("persona", "")
                        if persona not in reflections:
                            reflections[persona] = []
                        reflections[persona].append({
                            "step": rec.get("step"),
                            "insights": data.get("insights", []),
                        })

            summary = {
                "sim_code": self.sim_code,
                "start_time": self._start_wall,
                "end_time": end_wall,
                "total_steps": total_steps,
                "personas": personas,
                "timeline": timeline,
                "conversations": conversations,
                "daily_plans": daily_plans,
                "reflections": reflections,
            }
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# No-op logger (used before init_logger is called)
# ---------------------------------------------------------------------------
class _NoOpLogger:
    def set_step(self, *a, **kw): pass
    def log_step_complete(self, *a, **kw): pass
    def log_conversation(self, *a, **kw): pass
    def log_daily_plan(self, *a, **kw): pass
    def log_reflection(self, *a, **kw): pass
    def log_auto_save(self, *a, **kw): pass
    def log_gns_function(self, *a, **kw): pass
    def log_prompt(self, *a, **kw): pass
    def log_error(self, *a, **kw): pass
    def finalize(self, *a, **kw): pass
