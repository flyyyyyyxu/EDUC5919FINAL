"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling MiniMax APIs.
"""
import requests

from utils import *

MINIMAX_CHAT_COMPLETIONS_URL = f"{minimax_api_base}/chat/completions"


def ChatGPT_request(prompt):
  """
  Send a single prompt to MiniMax and return the text content.
  """
  try:
    response = requests.post(
      MINIMAX_CHAT_COMPLETIONS_URL,
      headers={
        "Authorization": f"Bearer {MiniMax_api_key}",
        "Content-Type": "application/json",
      },
      json={
        "model": minimax_text_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
      },
      timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
  except:
    print("MiniMax ERROR")
    return "MiniMax ERROR"


prompt = """
---
Character 1: Chen Zian is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context:
138 minutes ago, Chen Zian and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Chen Zian was attending her Physics class (preparing for the next lecture) when Chen Zian saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Chen Zian is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Chen Zian's head: Chen Zian should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Chen Zian doesn't necessarily know anything more about Klaus Mueller)

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Chen Zian about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Chen Zian)

Here is their conversation.

Chen Zian: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

print(ChatGPT_request(prompt))
