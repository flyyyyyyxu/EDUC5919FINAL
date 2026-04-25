"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling MiniMax text APIs.
"""
import hashlib
import json
import re
import time
from typing import Optional

_COMPLETION_SYSTEM = (
  "You are a text completion engine. The user gives you a text that ends "
  "mid-sentence, mid-phrase, or mid-list. Output ONLY the direct continuation "
  "of that text. Do not restate the prompt, do not explain, do not add "
  "any preamble or commentary. Just continue the text exactly where it left off."
)

# Patterns that indicate the model is narrating instead of completing
_PREAMBLE_RE = re.compile(
  r'^(the user\'?s?\b.{0,20}(asks?|says?|gives?|provides?|wants?|writes?|is asking|is telling|instruction)'
  r'|here (is|are) (my|the|a)\b'
  r'|i (should|need to|will|can|think|would)\b'
  r'|looking at\b'
  r'|based on\b'
  r'|so (i|the|we)\b'
  r'|thus\b'
  r'|therefore\b'
  r'|answer:\s*'
  r'|completion:\s*)',
  re.IGNORECASE,
)

import numpy
import requests

from utils import *
from json_logger import get_logger

MINIMAX_CHAT_COMPLETIONS_URL = f"{minimax_api_base}/chat/completions"
MINIMAX_LEGACY_COMPLETIONS_URL = f"{minimax_api_base}/text/chatcompletion_v2"
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_FALLBACK_STATUS_CODES = {400, 404, 405, 415, 422}
_REQUEST_TIMEOUT_SECONDS = 180
_MAX_COMPLETION_TOKENS_LIMIT = 2048


class MiniMaxConfigurationError(RuntimeError):
  pass


class MiniMaxAuthError(RuntimeError):
  pass


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _clamp_temperature(temperature):
  if temperature is None:
    return 1.0
  return min(max(float(temperature), 0.01), 1.0)


def _resolve_api_key():
  if not MiniMax_api_key:
    raise MiniMaxConfigurationError(
      "MiniMax API key is missing. Set MINIMAX_API_KEY, MINIMAX_KEY, "
      "or OPENAI_API_KEY before running the simulation."
    )
  return MiniMax_api_key


def _clamp_max_completion_tokens(max_tokens):
  tokens = int(max_tokens) if max_tokens is not None else 512
  tokens = max(1, tokens)
  return min(tokens, _MAX_COMPLETION_TOKENS_LIMIT)


def _extract_response_text(data):
  try:
    content = data["choices"][0]["message"]["content"]
  except (KeyError, IndexError, TypeError):
    raise RuntimeError(f"MiniMax response missing message content: {data}")
  if content is None:
    return ""
  return str(content)


def _should_fallback_to_legacy(status_code, body_text):
  if status_code not in _FALLBACK_STATUS_CODES:
    return False
  lowered = (body_text or "").lower()
  fallback_markers = (
    "max_completion_tokens",
    "unknown field",
    "invalid request",
    "not found",
    "unsupported media type",
  )
  return any(marker in lowered for marker in fallback_markers)


def _log_http_error(context, status_code, body_text, url):
  try:
    get_logger().log_error(
      context,
      f"HTTP {status_code} @ {url}: {body_text}",
    )
  except Exception:
    pass


def _post_minimax(url, payload, timeout):
  return requests.post(
    url,
    headers={
      "Authorization": f"Bearer {_resolve_api_key()}",
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    json=payload,
    timeout=timeout,
  )


def _request_with_retries(url,
                          payload,
                          timeout,
                          context,
                          max_attempts=3):
  last_error: Optional[Exception] = None
  for attempt in range(1, max_attempts + 1):
    try:
      response = _post_minimax(url, payload, timeout)
    except requests.RequestException as exc:
      last_error = exc
      if attempt == max_attempts:
        break
      time.sleep(min(2 ** (attempt - 1), 4))
      continue

    if response.ok:
      return response

    _log_http_error(context, response.status_code, response.text, url)
    if response.status_code in {401, 403}:
      raise MiniMaxAuthError(
        f"MiniMax authentication failed ({response.status_code}) for {url}. "
        "Check the API key and base URL in your environment."
      )
    if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
      time.sleep(min(2 ** (attempt - 1), 4))
      continue
    return response

  if last_error is not None:
    raise last_error
  raise RuntimeError(f"MiniMax request failed without response for {url}")


def _call_minimax(messages,
                  max_tokens=512,
                  temperature=1.0,
                  top_p=0.95,
                  model=None):
  # Prepend system message if not already present
  if not messages or messages[0].get("role") != "system":
    messages = [{"role": "system", "content": _COMPLETION_SYSTEM}] + messages

  max_completion_tokens = _clamp_max_completion_tokens(max_tokens)
  model_name = model or minimax_text_model
  payload = {
    "model": model or minimax_text_model,
    "messages": messages,
    "temperature": _clamp_temperature(temperature),
    "top_p": min(max(float(top_p), 0.01), 1.0),
    "max_completion_tokens": max_completion_tokens,
  }
  response = _request_with_retries(
    MINIMAX_CHAT_COMPLETIONS_URL,
    payload,
    _REQUEST_TIMEOUT_SECONDS,
    "minimax_http",
  )

  if (not response.ok and
      _should_fallback_to_legacy(response.status_code, response.text)):
    legacy_payload = {
      "model": model_name,
      "messages": messages,
      "temperature": _clamp_temperature(temperature),
      "top_p": min(max(float(top_p), 0.01), 1.0),
      "max_tokens": max_completion_tokens,
    }
    response = _request_with_retries(
      MINIMAX_LEGACY_COMPLETIONS_URL,
      legacy_payload,
      _REQUEST_TIMEOUT_SECONDS,
      "minimax_http_legacy",
    )

  response.raise_for_status()
  data = response.json()
  content = _extract_response_text(data)
  # Strip chain-of-thought think blocks
  content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
  # Drop leading lines that are meta-commentary rather than the actual completion
  lines = content.split("\n")
  for i, line in enumerate(lines):
    if line.strip() and not _PREAMBLE_RE.match(line.strip()):
      content = "\n".join(lines[i:]).strip()
      break
  return content


def ChatGPT_single_request(prompt):
  temp_sleep()
  return _call_minimax(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512,
    temperature=0.7,
  )


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to the
  MiniMax server and return the response.
  """
  temp_sleep()

  try:
    return _call_minimax(
      messages=[{"role": "user", "content": prompt}],
      max_tokens=1024,
      temperature=0.7,
      model=minimax_text_model,
    )
  except Exception as e:
    try:
      get_logger().log_error("gpt4_request", str(e))
    except Exception:
      pass
    if isinstance(e, (MiniMaxAuthError, MiniMaxConfigurationError)):
      raise
    return "MiniMax ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to the
  MiniMax server and return the response.
  """
  try:
    return _call_minimax(
      messages=[{"role": "user", "content": prompt}],
      max_tokens=1024,
      temperature=0.7,
      model=minimax_text_model,
    )
  except Exception as e:
    try:
      get_logger().log_error("chatgpt_request", str(e))
    except Exception:
      pass
    if isinstance(e, (MiniMaxAuthError, MiniMaxConfigurationError)):
      raise
    return "MiniMax ERROR"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  for i in range(repeat):
    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

    except (MiniMaxAuthError, MiniMaxConfigurationError):
      raise
    except:
      pass

  return fail_safe_response


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

    except (MiniMaxAuthError, MiniMaxConfigurationError):
      raise
    except:
      pass

  return fail_safe_response


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print(f"---- repeat count: {i}")
        print(curr_gpt_response)
        print("~~~~")
    except (MiniMaxAuthError, MiniMaxConfigurationError):
      raise
    except:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

_LIST_ITEM_RE = re.compile(r"(\d+\)\s*)$")

def GPT_request(prompt, gpt_parameter):
  """
  Legacy completion wrapper now routed through MiniMax chat completions.
  """
  temp_sleep()
  try:
    result = _call_minimax(
      messages=[{"role": "user", "content": prompt}],
      max_tokens=gpt_parameter.get("max_tokens", 512),
      temperature=gpt_parameter.get("temperature", 1.0),
      top_p=gpt_parameter.get("top_p", 0.95),
      model=gpt_parameter.get("model", minimax_text_model),
    )
    # If the prompt already ends with "N) " and the model repeated it, strip it
    m = _LIST_ITEM_RE.search(prompt.rstrip())
    if m and result.startswith(m.group(1)):
      result = result[len(m.group(1)):]
    return result
  except Exception as e:
    try:
      get_logger().log_error("gpt_request", str(e))
    except Exception:
      pass
    if isinstance(e, (MiniMaxAuthError, MiniMaxConfigurationError)):
      raise
    return "MINIMAX REQUEST FAILED"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input and a prompt template file, then substitutes the
  placeholders to produce the final prompt.
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = GPT_request(prompt, gpt_parameter)
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except (MiniMaxAuthError, MiniMaxConfigurationError):
      raise
  return fail_safe_response


def get_embedding(text, model="MiniMax-local-hash-embedding"):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  tokens = re.findall(r"\w+", text.lower())
  if not tokens:
    tokens = ["blank"]

  dim = 256
  vector = numpy.zeros(dim)
  for token in tokens:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    for idx in range(0, 32, 2):
      bucket = digest[idx] % dim
      sign = 1 if digest[idx + 1] % 2 == 0 else -1
      vector[bucket] += sign

  norm = numpy.linalg.norm(vector)
  if norm == 0:
    vector[0] = 1.0
    norm = numpy.linalg.norm(vector)
  return (vector / norm).tolist()


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True

  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt,
                                  gpt_parameter,
                                  5,
                                  "rest",
                                  __func_validate,
                                  __func_clean_up,
                                  True)

  print(output)
