"""보고서 영어판용 번역. 저장되는 결과는 한국어 그대로 두고, 표시할 때만 바꾼다.

사전은 화면과 함께 쓰는 static/i18n/en.json 하나: text는 한국어 원문 → 영어, patterns는 서버가 값을 끼워 만든 문장용.
사전에 없는 문장은 한국어 그대로 나온다.
"""
import json
import re
from pathlib import Path

_EN = json.loads((Path(__file__).parent / "static/i18n/en.json").read_text(encoding="utf-8"))
_PAT = [(re.compile("^" + re.sub(r"\\\{\d\\\}", "(.*)", re.escape(k)) + "$"), v) for k, v in _EN["patterns"].items()]
LANGS = ("ko", "en")


def translator(lang: str):
    def t(s: str, *args) -> str:
        out = s
        if lang == "en":
            out = _EN["text"].get(s, s)
            if out == s and not args:
                for rx, v in _PAT:
                    m = rx.match(s)
                    if m:
                        return v.format(*(t(g) for g in m.groups()))
        return out.format(*args) if args else out
    return t
