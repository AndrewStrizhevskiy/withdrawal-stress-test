"""Cycle 8 (protocol Section 46): answer parser for the Russian-language run, amended after the 60-item format checks.
1. Strip leading markdown and answer-announcing phrases, then read the first character: Latin A-D or Cyrillic
   А, В, С, Д (options 0-3), not followed by another letter.
2. Fallback: the first option letter that directly follows "ответ"/"answer" (optionally "на вопрос", a colon or dash,
   markdown) anywhere in the completion.
Otherwise invalid (-1).
"""
import re

MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "А": 0, "В": 1, "С": 2, "Д": 3}
PREFIX = re.compile(r"^(?:the\s+correct\s+answer\s+is|correct\s+answer|правильный\s+ответ(?:\s+на\s+(?:этот\s+)?вопрос)?|"
                    r"буква\s+правильного\s+ответа|ответ\s+на\s+(?:этот\s+)?вопрос|ответ|answer|буква|вариант)"
                    r"\s*(?:[:\-—–]|это|is)?\s*", re.I)
JUNK = re.compile(r"^[\s*#_`\"'«(\[]+")
FALLBACK = re.compile(r"(?i:ответ|answer)(?i:\s+на\s+(?:этот\s+)?вопрос)?\s*(?i:[:\-—–]|это|is)?\s*[*_\"'«(\[\s]*"
                      r"([ABCDАВСД])(?![A-Za-zА-Яа-яЁё])")   # option letter itself is case-sensitive


def parse(comp):
    t = comp.split("</think>", 1)[1] if "</think>" in comp else comp
    s = t
    for _ in range(3):
        s = JUNK.sub("", s)
        s2 = PREFIX.sub("", s)
        if s2 == s:
            break
        s = s2
    s = JUNK.sub("", s)
    if s and s[0] in MAP and (len(s) == 1 or not s[1].isalpha()):
        word_follows = len(s) > 2 and s[1] == " " and s[2].islower()      # "В данном случае ..." is a preposition
        if not word_follows:
            return MAP[s[0]]
    m = FALLBACK.search(t)
    return MAP[m.group(1)] if m else -1
