"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling MiniMax text APIs.
"""
import hashlib
import json
import re
import time

import numpy
import requests

from utils import *

MINIMAX_CHAT_COMPLETIONS_URL = f"{minimax_api_base}/chat/completions"


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _clamp_temperature(temperature):
  if temperature is None:
    return 1.0
  return min(max(float(temperature), 0.01), 1.0)


def _call_minimax(messages,
                  max_tokens=512,
                  temperature=1.0,
                  top_p=0.95,
                  model=None):
  payload = {
    "model": model or minimax_text_model,
    "messages": messages,
    "temperature": _clamp_temperature(temperature),
    "top_p": min(max(float(top_p), 0.01), 1.0),
    "max_tokens": int(max_tokens),
  }
  response = requests.post(
    MINIMAX_CHAT_COMPLETIONS_URL,
    headers={
      "Authorization": f"Bearer {MiniMax_api_key}",
      "Content-Type": "application/json",
    },
    json=payload,
    timeout=180,
  )
  response.raise_for_status()
  data = response.json()
  return data["choices"][0]["message"]["content"]


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
  except:
    print("MiniMax ERROR")
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
  except:
    print("MiniMax ERROR")
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

  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")
    except:
      pass

  return False


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

  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")
    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
  if verbose:
    print("CHAT GPT PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print(f"---- repeat count: {i}")
        print(curr_gpt_response)
        print("~~~~")
    except:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """
  Legacy completion wrapper now routed through MiniMax chat completions.
  """
  temp_sleep()
  try:
    return _call_minimax(
      messages=[{"role": "user", "content": prompt}],
      max_tokens=gpt_parameter.get("max_tokens", 512),
      temperature=gpt_parameter.get("temperature", 1.0),
      top_p=gpt_parameter.get("top_p", 0.95),
      model=gpt_parameter.get("model", minimax_text_model),
    )
  except:
    print("MINIMAX REQUEST FAILED")
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
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print("---- repeat count: ", i, curr_gpt_response)
      print(curr_gpt_response)
      print("~~~~")
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
