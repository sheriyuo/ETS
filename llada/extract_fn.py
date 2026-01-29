import textwrap
import re
from typing import List, Optional

from lm_eval.tasks.minerva_math.utils import get_unnormalized_answer, normalize_final_answer
# from lm_eval.tasks.humaneval.utils import 

def gsm8k_extract(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    # if len(solution_str) > _SOLUTION_CLIP_CHARS:
    #     solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(-?[$0-9.,]{2,})|(-?[0-9]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def humaneval_extract(text: str, entry_point: str):
    """
    从 text 中提取被 ``` ``` 包裹的代码块，
    返回包含 `def {entry_point}(...):` 的那个代码块（不含 ```）。
    若不存在则返回 None。
    """
    code_prefix = text.split("```", 1)[0]
    return code_prefix.rstrip()
    # codeblock_re = re.compile(r"```[^\n]*\n(.*?)(?:\n)?```", flags=re.DOTALL)
    # codeblocks = codeblock_re.findall(text)

    # # def entry_point pattern (line-anchored)
    # func_re = re.compile(
    #     rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(.*?\)\s*:",
    #     flags=re.DOTALL
    # )

    # # 从后往前找：返回最后一个命中的 fenced block
    # for code in reversed(codeblocks):
    #     if func_re.search(code):
    #         return code

    # # --- step 2: fallback: search in full text from the end ---
    # # 找全文里最后一个 def entry_point 的位置
    # all_matches = list(func_re.finditer(text))
    # if not all_matches:
    #     return None

    # m = all_matches[-1]
    # start = m.start()

    # # 启发式截取：从该 def 行开始，到末尾；并尝试去掉尾部的多余围栏/解释文字
    # tail = text[start:]

    # # 如果 tail 里后面还出现了 ```，通常意味着又开始了新的 fenced 段或分隔符；
    # # 这里取 def 之后到下一个 ``` 之前的内容，避免带入后续非代码说明。
    # fence_pos = tail.find("```")
    # if fence_pos != -1:
    #     tail = tail[:fence_pos]

    # # 去掉可能的结尾多余空白
    # return tail.strip()

    # code_block_pattern = re.compile(
    #     rf"```(?:[Pp]ython\n)?[\s\S]*?def\s+{entry_point}.*?:\n(?:(?!```).)*```",
    #     re.DOTALL
    # )

    # code_block = code_block_pattern.search(text)
    # print(code_block)
    # if code_block is None:
    #     code_block_pattern = re.compile(
    #         rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
    #     )
    #     code_block = code_block_pattern.search(text)

    # if code_block is None:
    #     code_block_pattern = re.compile(
    #         r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
    #     )
    #     code_block = code_block_pattern.search(text)

    # if code_block is not None:
    #     return code_block.group(1)

    # # if no code block is found, assume the LM is simply filling the code
    # return textwrap.indent(text, " " * 4)

def math_extract(x):
    return normalize_final_answer(get_unnormalized_answer(x))

def parse_answer_gpqa(pred: str):
    pred = pred.replace("\u043a\u0438", "")
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred

if __name__ == "__main__":
#     code = '''     if len(s0) != len(s1):                                     
#         return False                                                                          
#     for i in range(len(s0)):
#         if s0[i] != s1[i]:
#             return False
#     return True

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
# ```

# ### Explanation:
# 1. **Function Definition**: The function `same_chars` takes two strings `s0` and `s1` as input
# .
# 2. **Length Check**: It first checks if the lengths of the two strings are different. If they 
# are, it returns `False` because strings of different lengths cannot have the same characters.
# 3. **Character Comparison**: It then iterates through the characters of both strings. If any c
# haracter at the same position is different, it returns `False`.
# 4. **Return True**: If all characters are the same, it returns `True`.
# 5. **Test Cases**: The function includes several test cases to verify its correctness.

# ### Running the Code:
# To run the code, simply execute it in a Python environment. The `doctest` module will automati
# cally run the test cases and display the results.

# ```python
# def same_chars(s0: str, s1: str):
#     """
#     Check if two words have the same characters.
#     >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
#     True
#     >>> same_chars('abcd', 'dddddddabc')
#     True
#     >>> same_chars('dddddddabc', 'abcd')
#     True
#     >>> same_chars('eabcd', 'dddddddabc')
#     False
#     >>> same_chars('abcd', 'dddddddabce')
#     False
#     >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
#     False
#     """
#     if len(s0) != len(s1):
#         return False
#     for i in range(len(s0)):
#         if s0[i] != s1[i]:
#             return False
#     return True 

# if __name__ == "__main__":                              
#     import doctest          
#     doctest.testmod()       
# ```                         

# This code will execute the test cases and provide the output.<|eot_id|><|endoftext|><|endoftext|><|endoftext|><|
# endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftex
# t|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|end
# oftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>          
# '''

#     entry_point = "same_chars"

#     extract_code = humaneval_extract(code, entry_point)
#     print(extract_code)
    ans1 = "First find the total cost of the green apple gum: $2/pack / 2 = $<<2/2=1>>1\nThen find the total cost of the grape gum: $2/pack * 1 pack = $<<2*1=2>>2\nThen find the total cost of the strawberry gum by subtracting the cost of the grape and green apple gum from the total cost: $7 - $2 - $1 = $<<7-2-1=4>>4\nThen find the cost of each pack of strawberry gum by dividing the total cost by the number of packs: $4 / 2 packs = $<<4/2=2>>2/pack\n#### 2"
    ans2 = "To find out how many medals have been given out for counting, we need to follow these steps:\n\n1. Determine how many medals Izzy has.\n2. Calculate the total number of medals Ali and Izzy have together.\n3. Use the information that together they have 10 times less medals than have been given out to find the total number of medals given out.\n\nFirst, we know Ali has 22 medals. Izzy has 5 less medals than Ali, so Izzy has:\n\\[ 22 - 5 = 17 \\text{ medals} \\]\n\nNext, we calculate the total number of medals Ali and Izzy have together:\n\\[ 22 + 17 = 39 \\text{ medals} \\]\n\nWe are told that together they have 10 times less medals than have been given out. This means that the total number of medals given out is 10 times the number of medals they have. Therefore, the total number of medals given out is:\n\\[ 39 \\times 10 = 390 \\]\n\nSo, the total number of medals given out for counting is 390."
    ans3 = "To determine the total points Sasha scored during both games, we need to follow these steps:\n\n1. Calculate Julie's score in the first game.\n2. Calculate Sasha's score in the second game.\n3. Add Sasha's scores from both games to get the total.\n\nFirst, let's calculate Julie's score in the first game:\n- Sasha scored 14 points in the first game.\n- Julie scored 4 fewer points than Sasha.\n- Therefore, Julie scored \\( 14 - 4 = 10 \\) points.\n\nNext, let's calculate Sasha's score in the second game:\n- Sasha scored 6 fewer points in the second game than Julie's score in the first game.\n- Julie scored 10 points in the first game.\n- Therefore, Sasha scored \\( 10 - 6 = 6 \\) points in the second game.\n\nNow, let's add Sasha's scores from both games:\n- Sasha scored 14 points in the first game.\n- Sasha scored 6 points in the second game.\n- Total points Sasha scored: \\( 14 + 6 = 20 \\) points.\n\n#### 20"
    extract_answer1 = gsm8k_extract(ans1, method="flexible")
    extract_answer2 = gsm8k_extract(ans2, method="flexible")
    extract_answer3 = gsm8k_extract(ans3, method="flexible")
    if extract_answer1[0] == "":
        if extract_answer1[1] != "":
            extract_answer1 = extract_answer1[1]
        else:
            extract_answer1 = None
    else:
        extract_answer1 = extract_answer1[0]
    print(extract_answer1)
    print(extract_answer2)
    print(extract_answer3)