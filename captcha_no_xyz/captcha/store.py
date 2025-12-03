import threading
import time
from typing import Dict, Tuple, Optional, Any


class CaptchaStore:
    def __init__(self):
        self._lock = threading.Lock()
        # id -> { answer, expires_at, norm, kind?, prompt?, meta? }
        self._data: Dict[str, Dict[str, Any]] = {}

    def put(
        self,
        id_: str,
        answer: str,
        expires_at: float,
        norm: str = "lower",
        kind: Optional[str] = None,
        prompt: Optional[str] = None,
        meta: Optional[dict] = None,
        svg: Optional[str] = None,
        data_uri: Optional[str] = None,
    ):
        with self._lock:
            self._data[id_] = {
                "answer": answer,
                "expires_at": float(expires_at),
                "norm": norm,
                "kind": kind,
                "prompt": prompt,
                "meta": meta,
                "svg": svg,
                "data_uri": data_uri,
            }

    def verify(self, id_: str, user_answer: str) -> tuple[bool, str]:
        now = time.time()
        with self._lock:
            if id_ not in self._data:
                return False, "not_found"
            entry = self._data[id_]
            answer = entry.get("answer", "")
            # exp = entry.get("expires_at", 0.0)
            # if float(exp) < now:
            #     del self._data[id_]
            #     return False, "expired"
            norm = entry.get("norm", "exact")
            kind = (entry.get("kind") or "").lower()

            ua_raw = (user_answer or "").strip()

            # anagram：接受任意由相同字母组成的有效单词；若有候选列表则优先判断是否在列表中
            if kind == "anagram":
                ua_word = ua_raw.lower()
                try:
                    meta = entry.get("meta") or {}
                    words = meta.get("words") or []
                    words_lower = {str(w).lower() for w in words if isinstance(w, str)}
                except Exception:
                    words_lower = set()

                def _letters_signature(s: str) -> str:
                    letters = [ch.lower() for ch in s if ch.isalpha()]
                    letters.sort()
                    return "".join(letters)

                if words_lower and ua_word in words_lower:
                    ok = True
                else:
                    ok = _letters_signature(ua_word) == _letters_signature(answer)
                if ok:
                    del self._data[id_]
                    return True, "ok"
                return False, "mismatch"

            if norm == "exact":
                ok = (ua_raw == answer)
            elif norm == "lower":
                ok = (ua_raw.lower() == answer)
            elif norm == "int":
                try:
                    ok = (int(ua_raw) == int(answer))
                except Exception:
                    return False, "invalid_format"
            elif norm.startswith("point:"):
                try:
                    tol = float(norm.split(":", 1)[1])
                    ax, ay = [float(v) for v in answer.split(",", 1)]
                    ux, uy = [float(v) for v in ua_raw.split(",", 1)]
                except Exception:
                    return False, "invalid_format"
                # Primary check (assume same origin)
                dx = ux - ax
                dy = uy - ay
                ok = (dx * dx + dy * dy) ** 0.5 <= tol
                if not ok:
                    # Fallback: tolerate top-left/bottom-left origin mismatch using meta.height
                    try:
                        h = float((entry.get("meta") or {}).get("height") or 0)
                    except Exception:
                        h = 0
                    if h:
                        candidates = [
                            (ux, h - uy, ax, ay),       # user bl, stored tl
                            (ux, uy, ax, h - ay),       # user tl, stored bl
                            (ux, h - uy, ax, h - ay),   # both flipped
                        ]
                        for uxo, uyo, axo, ayo in candidates:
                            dx = uxo - axo
                            dy = uyo - ayo
                            if (dx * dx + dy * dy) ** 0.5 <= tol:
                                ok = True
                                break
            elif norm.startswith("approx:"):
                try:
                    tol = int(float(norm.split(":", 1)[1]))
                    ok = abs(int(ua_raw) - int(answer)) <= tol
                except Exception:
                    return False, "invalid_format"
            elif norm == "set":
                # compare as set of integers (order-insensitive). Accept empty => empty
                try:
                    a_set = set(int(x) for x in filter(None, (s.strip() for s in answer.split(','))))
                    u_set = set(int(x) for x in filter(None, (s.strip() for s in ua_raw.split(','))))
                except Exception:
                    return False, "invalid_format"
                ok = (a_set == u_set)
            elif norm == "list":
                # compare as ordered list of integers
                try:
                    a_list = [int(x) for x in filter(None, (s.strip() for s in answer.split(',')))]
                    u_list = [int(x) for x in filter(None, (s.strip() for s in ua_raw.split(',')))]
                except Exception:
                    return False, "invalid_format"
                ok = (a_list == u_list)
            else:
                ok = (ua_raw == answer)
            # one-time use: remove on success
            if ok:
                del self._data[id_]
                return True, "ok"
            else:
                return False, "mismatch"

    def cleanup(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for k in list(self._data.keys()):
                exp = self._data[k].get("expires_at", 0.0)
                if float(exp) < now:
                    del self._data[k]
                    removed += 1
        return removed

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._data.get(id_, {})) if id_ in self._data else None

    def update(self, id_: str, **kwargs) -> bool:
        with self._lock:
            if id_ not in self._data:
                return False
            self._data[id_].update(kwargs)
            return True
