import sys
import os
import pytest
import re

def get_funcs():
    file_path = os.path.join(os.path.dirname(__file__), "..", "training.py")
    with open(file_path, "r") as f:
        content = f.read()
    
    # Extract format_reward_func
    match_format = re.search(r"def format_reward_func.*?return rewards\n", content, re.DOTALL)
    match_parse = re.search(r"def parse_action.*?return None, None\n", content, re.DOTALL)
    
    namespace = {}
    exec("import re\n" + match_parse.group(0) + "\n" + match_format.group(0), globals(), namespace)
    return namespace["format_reward_func"], namespace["parse_action"]

def test_format_reward_func():
    format_reward_func, _ = get_funcs()
    completions = [
        "Just some text without tags.",
        "Here is a <thought>I should check logs</thought>.",
        "<action>read_logs:payment-service</action>",
        "<thought>Thinking</thought> then <action>fix:api</action>"
    ]
    prompts = [""] * len(completions)
    
    rewards = format_reward_func(prompts, completions)
    
    # 1. No tags = 0.0
    assert rewards[0] == 0.0
    # 2. Only thought tag = 0.1
    assert rewards[1] == 0.1
    # 3. Only action tag = 0.1
    assert rewards[2] == 0.1
    # 4. Both tags = 0.2
    assert rewards[3] == 0.2

def test_parse_action():
    _, parse_action = get_funcs()
    atype, target = parse_action("Some text <action>read_logs:user-service</action> other text")
    assert atype == "read_logs"
    assert target == "user-service"

if __name__ == "__main__":
    test_format_reward_func()
    test_parse_action()
    print("Tests passed successfully!")
