"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: print_prompt.py
Description: For printing prompts when the setting for verbose is set to True.
"""
import sys
sys.path.append('../')

import json
import numpy
import datetime
import random

from global_methods import *
from persona.prompt_template.gpt_structure import *
from utils import *

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                    #
##############################################################################

def print_run_prompts(prompt_template=None, 
                      persona=None, 
                      gpt_param=None, 
                      prompt_input=None,
                      prompt=None, 
                      output=None): 
  try:
    from json_logger import get_logger
    get_logger().log_prompt(
      template=str(prompt_template),
      persona_name=persona.name if persona else "",
      gpt_param=gpt_param or {},
      prompt_input=prompt_input,
      prompt=str(prompt),
      output=output,
    )
  except Exception:
    pass
