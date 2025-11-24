import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import os
import re
from typing import Optional, List, Tuple, Union, Dict, Any

from .generators import CaptchaFactory
from .store import CaptchaStore
from .agents import IdentificationAgent, ExecutionAgent, AgentResult, CaptchaData

_STORE = CaptchaStore()
_FACTORY = CaptchaFactory()
_ID_AGENT = IdentificationAgent()
_EXEC_AGENT = ExecutionAgent()
_TTL_SECONDS = 0  # 0 means never expire
_DEBUG = False


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _ensure_image_size(info: dict) -> tuple[int, int]:
    """Try best-effort to obtain (image_width, image_height).

    Priority:
      1) explicit image_width/image_height in info
      2) parse from data_uri (embedded SVG)
      3) fallback to meta.width/meta.height
    """
    try:
        w = int(float(info.get("image_width") or 0))
        h = int(float(info.get("image_height") or 0))
    except Exception:
        w = h = 0
    if w and h:
        return w, h
    # Parse from data_uri (SVG)
    try:
        data_uri = info.get("data_uri") or ""
        if isinstance(data_uri, str) and data_uri.startswith("data:image/svg+xml"):
            import base64, urllib.parse
            payload = data_uri.split(",", 1)[1] if "," in data_uri else ""
            # handle base64 or urlencoded
            if ";base64" in data_uri:
                svg = base64.b64decode(payload).decode("utf-8", errors="ignore")
            else:
                svg = urllib.parse.unquote(payload)
            # viewBox first
            m = re.search(r"viewBox\s*=\s*['\"]\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*['\"]", svg, re.I)
            if m:
                w = w or int(float(m.group(1)))
                h = h or int(float(m.group(2)))
            else:
                m2 = re.search(r"width=['\"]([0-9.]+)['\"][^>]*height=['\"]([0-9.]+)['\"]", svg, re.I)
                if m2:
                    w = w or int(float(m2.group(1)))
                    h = h or int(float(m2.group(2)))
            if w and h:
                return w, h
    except Exception:
        pass
    # Fallback to meta
    try:
        meta = info.get("meta") or {}
        w = w or int(float(meta.get("width") or 0))
        h = h or int(float(meta.get("height") or 0))
    except Exception:
        pass
    return int(w or 0), int(h or 0)


def _load_image_from_info(info: dict):
    """Best-effort decode image from image_png or data_uri; returns PIL Image or None."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    import base64, io
    data_uri = info.get("image_png") or info.get("data_uri")
    if not isinstance(data_uri, str) or "," not in data_uri:
        return None
    try:
        b64 = data_uri.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _gridcolor_detect_indices(info: dict) -> Optional[List[int]]:
    """Deterministically pick color-only targets using palette + image sampling (gridcolor only)."""
    try:
        kind = (info.get("kind") or info.get("type") or "").lower()
        if kind != "gridcolor":
            return None
        palette = []
        for item in (info.get("meta") or {}).get("palette", []):
            n = item.get("name")
            rgb = item.get("rgb")
            if isinstance(n, str) and isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
                try:
                    palette.append((n, (int(rgb[0]), int(rgb[1]), int(rgb[2]))))
                except Exception:
                    continue
        if not palette:
            return None
        prompt = str(info.get("prompt") or "")
        target_name = None
        for n, _ in palette:
            if n in prompt:
                target_name = n
                break
        if not target_name:
            return None
        target_rgb = None
        other = []
        for n, rgb in palette:
            if n == target_name:
                target_rgb = rgb
            else:
                other.append((n, rgb))
        if target_rgb is None:
            return None
        img = _load_image_from_info(info)
        if img is None:
            return None
        w = img.width
        h = img.height
        rows = int(((info.get("meta") or {}).get("rows") or 3))
        cols = int(((info.get("meta") or {}).get("cols") or 3))
        def dist2(c1, c2):
            return sum((int(c1[i]) - int(c2[i])) ** 2 for i in range(3))
        result = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx = min(w - 1, max(0, int((c + 0.5) * w / cols)))
                cy = min(h - 1, max(0, int((r + 0.5) * h / rows)))
                pix = img.getpixel((cx, cy))
                d_target = dist2(pix, target_rgb)
                best_other = min(dist2(pix, rgb) for _, rgb in other) if other else 1e9
                # 要求颜色比任何其他调色板颜色更接近目标，且差距足够明显
                if d_target < best_other and (best_other - d_target) >= 500:  # margin
                    result.append(idx)
        return sorted(set(result)) if result else []
    except Exception:
        return None


def _scrub_info(info: dict) -> dict:
    """Remove answer/secret fields before sending to LLM."""
    try:
        if not isinstance(info, dict):
            return {}
        out = {}
        for k, v in info.items():
            # drop anything that明显包含答案或过期时间
            if k in {"answer", "expires_at", "debug_answer"}:
                continue
            if "answer" in k.lower():
                continue
            if isinstance(v, dict):
                out[k] = _scrub_info(v)
            else:
                out[k] = v
        return out
    except Exception:
        return {}


def _iter_strings(obj):
    """Yield string fragments from nested dict/list structures."""
    try:
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    yield k
                yield from _iter_strings(v)
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                yield from _iter_strings(it)
    except Exception:
        return


def _try_extract_click_xy(reco: dict, w: int, h: int) -> Optional[Tuple[int, int]]:
    """Best-effort extract a bottom-left-origin click (x,y) from reco JSON.

    Accepts patterns:
      - fields: {"x":123, "y":45}
      - point dict: {"point": {"x":123, "y":45}} 或 {"calc_center": {...}}
      - 自定义标记: {"标记坐标": [{"x":..,"y":..}, ...]}
      - text like: "point 123,45" / "坐标: 123,45" / "x=123, y=45"
    Returns clamped integers within [0..w],[0..h] or None if not found.
    """
    # 标记坐标数组: 取第一个点
    try:
        marks = reco.get("标记坐标") if isinstance(reco, dict) else None
        if isinstance(marks, list) and marks:
            pt = marks[0]
            if isinstance(pt, dict) and isinstance(pt.get("x"), (int, float)) and isinstance(pt.get("y"), (int, float)):
                x = int(round(float(pt.get("x"))))
                y = int(round(float(pt.get("y"))))
                if w and h and (x < 0 or y < 0 or x > int(w) or y > int(h)):
                    return None
                return x, y
    except Exception:
        pass
    # Direct fields
    try:
        if isinstance(reco, dict) and isinstance(reco.get("x"), (int, float)) and isinstance(reco.get("y"), (int, float)):
            x = int(round(float(reco.get("x"))))
            y = int(round(float(reco.get("y"))))
            if w and h:
                # 若越界，说明使用了不同坐标系，直接放弃直提，交给下游 LLM 处理
                if x < 0 or y < 0 or x > int(w) or y > int(h):
                    return None
            return x, y
        # nested point / calc_center
        for key in ("point", "calc_center"):
            pt = reco.get(key) if isinstance(reco, dict) else None
            if (
                isinstance(pt, dict)
                and isinstance(pt.get("x"), (int, float))
                and isinstance(pt.get("y"), (int, float))
            ):
                x = int(round(float(pt.get("x"))))
                y = int(round(float(pt.get("y"))))
                if w and h:
                    if x < 0 or y < 0 or x > int(w) or y > int(h):
                        return None
                return x, y
    except Exception:
        pass
    # Scan text
    try:
        for s in _iter_strings(reco):
            if not isinstance(s, str) or len(s) > 4000:
                continue
            m = re.search(r"(?:point|坐标)\s*[:：]?\s*([0-9]+)\s*,\s*([0-9]+)", s, re.I)
            if not m:
                m = re.search(r"x\s*[:=]\s*([0-9]+)\s*,?\s*y\s*[:=]\s*([0-9]+)", s, re.I)
            if m:
                x = int(m.group(1))
                y = int(m.group(2))
                if w and h:
                    if x < 0 or y < 0 or x > int(w) or y > int(h):
                        return None
                return x, y
    except Exception:
        pass
    return None


def _try_extract_input_value(info: dict, reco: dict) -> Optional[str]:
    """Try to deterministically extract an input value from recognition output.

    Heuristics:
      1) reco.value / reco.answer if present
      2) From steps: prefer label in {正确答案, 答案, 最终答案, 输入值}
         - pick the longest alnum/arrow/compare token (e.g. jjgqr, 123, >, YES)
      3) Fallback: scan all strings in reco and pick a concise candidate

    Apply norm rules: lower/int/exact.
    """
    norm = (info.get("norm") or "exact").lower()

    def norm_val(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        if norm == "lower":
            return s.lower()
        if norm == "int":
            try:
                return str(int(float(s.strip())))
            except Exception:
                return None  # invalid
        # exact or others
        return s

    # 1) direct fields
    for k in ("value", "answer", "验证码为"):
        v = reco.get(k)
        if isinstance(v, str) and v.strip():
            nv = norm_val(v.strip())
            if nv is not None:
                return nv

    import re as _re
    preferred = {"正确答案", "答案", "最终答案", "输入值", "验证码为"}

    def extract_token(text: str) -> Optional[str]:
        # allow letters/digits/arrow symbols/compare signs
        m = _re.findall(r"[A-Za-z0-9><=]+", text)
        if not m:
            return None
        # 如果全部是单字符且有多个，按出现顺序拼接，适用于“3 4 p B W”这类 case 题
        if all(len(t) == 1 for t in m) and len(m) > 1:
            joined = "".join(m)
            nv = norm_val(joined)
            return nv
        # otherwise prefer the longest token
        cand = max(m, key=len)
        nv = norm_val(cand)
        return nv

    def extract_answer_token(text: str) -> Optional[str]:
        """针对“正确答案”类文案更保守地提取答案 token。

        规则：
          - 优先丢弃明显是元信息的 token，例如包含 'norm'/'type' 或带 '=' 的长串
          - 在候选集中，优先选择包含数字的 token（典型验证码如 mjm44），再按长度优先
          - 若只有单字符序列，则按顺序拼接（兼容逐字列出场景）
        """
        m = _re.findall(r"[A-Za-z0-9><=]+", text)
        if not m:
            return None
        cands: list[str] = []
        for t in m:
            tl = t.lower()
            # 过滤明显的元信息字段
            if "norm" in tl or "type" in tl:
                continue
            # 带 '=' 的一般是说明（如 norm=lower），但比较题的真正答案仅为单个符号 >/</=，已被下一条规则覆盖
            if "=" in t and tl not in {">", "<", "="}:
                continue
            cands.append(t)
        if not cands:
            return extract_token(text)
        # 如果全部是单字符且有多个，按出现顺序拼接
        if all(len(t) == 1 for t in cands) and len(cands) > 1:
            joined = "".join(cands)
            nv = norm_val(joined)
            return nv
        # 否则优先“包含数字”的 token，再按长度优先
        def _score(t: str) -> tuple[int, int]:
            has_digit = any(ch.isdigit() for ch in t)
            return (0 if has_digit else 1, -len(t))
        best = min(cands, key=_score)
        return norm_val(best)

    # 2) math-aware pass: if norm=int，尝试从描述中识别算式并计算/取等号后数值
    def _math_value(text: str) -> Optional[str]:
        try:
            # 优先取等号后的数字，如 "13 - 12 = 1"
            m_eq = _re.search(r"=\s*([0-9]+)", text)
            if m_eq:
                return str(int(m_eq.group(1)))
            m = _re.search(r"([0-9]+)\s*([+\-×x*/÷])\s*([0-9]+)", text)
            if not m:
                return None
            a = int(m.group(1)); b = int(m.group(3)); op = m.group(2)
            if op in {"+", "＋"}:
                return str(a + b)
            if op in {"-", "−"}:
                return str(a - b)
            if op in {"×", "x", "X", "*"}:
                return str(a * b)
            if op in {"÷", "/"}:
                if b != 0 and a % b == 0:
                    return str(a // b)
                if b != 0:
                    return str(a / b)
            return None
        except Exception:
            return None

    if norm == "int":
        steps = reco.get("steps")
        if isinstance(steps, list):
            for it in steps:
                try:
                    det = str((it or {}).get("detail") or "")
                    mv = _math_value(det)
                    if mv is not None:
                        return mv
                except Exception:
                    pass

    # 3) scan steps with preferred labels first
    steps = reco.get("steps")
    if isinstance(steps, list):
        # preferred labels
        for it in steps:
            try:
                lbl = str((it or {}).get("label") or "")
                if lbl in preferred:
                    det = str((it or {}).get("detail") or "")
                    tok = extract_answer_token(det)
                    if tok:
                        return tok
            except Exception:
                pass
        # any detail
        for it in steps:
            try:
                det = str((it or {}).get("detail") or "")
                tok = extract_token(det)
                if tok:
                    return tok
            except Exception:
                pass

    # 4) scan all strings
    for s in _iter_strings(reco):
        if not isinstance(s, str) or len(s) > 4000:
            continue
        # math 优先
        if norm == "int":
            mv = _math_value(s)
            if mv is not None:
                return mv
        tok = extract_token(s)
        if tok:
            return tok
    return None


def _try_extract_grid_indices(info: dict, reco: dict) -> Optional[List[int]]:
    """Attempt to extract full grid indices from recognition output without LLM.

    Sources considered:
      - reco.indices: [ints]（最可信，优先使用）
    """
    try:
        if not isinstance(reco, dict):
            return None
        kind = (info.get("kind") or info.get("type") or "").lower()
        # gridcolor：尝试直接从图像按 palette 计算颜色最近的格子，避免 LLM 误判
        if kind == "gridcolor":
            auto = _gridcolor_detect_indices(info)
            if auto:
                return auto
        # 0) direct indices come first
        inds = reco.get("indices")
        if isinstance(inds, list):
            out = []
            for v in inds:
                try:
                    out.append(int(v))
                except Exception:
                    pass
            if out:
                rows = int(((info.get("meta") or {}).get("rows") or 0))
                cols = int(((info.get("meta") or {}).get("cols") or 0))
                n = rows * cols
                # 若全部在 1..n 且无 0，视为 1 基索引，转换为 0 基
                if n and min(out) >= 1 and max(out) <= n and 0 not in out:
                    out = [i - 1 for i in out]
                return sorted(set(out))
    except Exception:
        return None
    return None


def _try_extract_seq_indices(info: dict, reco: dict) -> Optional[List[int]]:
    """Extract ordered indices for sequence tasks without LLM.

    只接受识别Agent直接给出的索引（indices 或 grid_indices），严格长度与唯一性校验。
    若未显式给出索引，则尝试从行列(row/col)或像素坐标列表推导（标记/执行坐标等）。
    """
    try:
        if not isinstance(reco, dict):
            return None
        k = int(((info.get("meta") or {}).get("k") or 0))
        inds = reco.get("indices")
        if not inds:
            inds = reco.get("grid_indices")
        if isinstance(inds, list) and inds:
            out: list[int] = []
            for v in inds:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            if out:
                rows = int(((info.get("meta") or {}).get("rows") or 0))
                cols = int(((info.get("meta") or {}).get("cols") or 0))
                n = rows * cols
                # 如果是 1 基索引，转换为 0 基
                if n and min(out) >= 1 and max(out) <= n and 0 not in out:
                    out = [i - 1 for i in out]
                # 若取到的数值明显超出格子范围（像素坐标），尝试按坐标对待
                if n and out and max(out) >= n and rows and cols:
                    out = []
                if k > 0 and (len(out) != k):
                    return None
                if out:
                    return out
            # indices 里放的是坐标或 row/col 时，后续统一处理
        # 若 indices/grid_indices 失败，尝试从其他字段或 indices 原内容解析
        # 行列信息转换为索引
        rows = int(((info.get("meta") or {}).get("rows") or 0))
        cols = int(((info.get("meta") or {}).get("cols") or 0))
        w, h = _ensure_image_size(info)
        # 继续尝试从其他字段推导索引（行列或像素坐标）
        cand_lists: Optional[list] = None
        # 先把 indices 原内容加入候选（可能是坐标对）
        if isinstance(inds, list) and inds:
            cand_lists = inds
        for key in ("标记坐标", "执行坐标", "points"):
            val = reco.get(key)
            if isinstance(val, list) and val:
                cand_lists = val
                break
        if cand_lists is None:
            pt = reco.get("point")
            if isinstance(pt, dict):
                cand_lists = [pt]
        if rows and cols and w and h and isinstance(cand_lists, list) and cand_lists:
            tmp: list[int] = []
            # 优先处理 row/col
            all_rc = all(isinstance(it, dict) and "row" in it and "col" in it for it in cand_lists)
            if all_rc:
                for it in cand_lists:
                    try:
                        r = int(it.get("row")); c = int(it.get("col"))
                        idx = (r - 1) * cols + (c - 1)  # 转为 0 基索引
                        if 0 <= idx < rows * cols:
                            tmp.append(idx)
                    except Exception:
                        continue
                if tmp and (k == 0 or len(tmp) == k):
                    return tmp
                tmp = []
            # 其次处理像素坐标（dict/list/tuple）
            def _convert(pts, use_top_origin: bool) -> tuple[list[int], float]:
                out: list[int] = []
                err = 0.0
                cell_w = w / cols
                cell_h = h / rows
                for it in pts:
                    try:
                        if isinstance(it, dict) and "x" in it and "y" in it:
                            px = float(it.get("x")); py = float(it.get("y"))
                        elif isinstance(it, (list, tuple)) and len(it) >= 2:
                            px = float(it[0]); py = float(it[1])
                        else:
                            continue
                        py_use = py if not use_top_origin else h - py
                        col = max(0, min(cols - 1, int((px / w) * cols)))
                        row = max(0, min(rows - 1, int(((h - py_use) / h) * rows)))
                        idx = row * cols + col  # 0 基索引
                        out.append(idx)
                        cx = (col + 0.5) * cell_w
                        cy = h - (row + 0.5) * cell_h
                        dy = py_use - cy
                        dx = px - cx
                        err += dx * dx + dy * dy
                    except Exception:
                        continue
                return out, err

            idx_bl, err_bl = _convert(cand_lists, False)
            idx_tl, err_tl = _convert(cand_lists, True)
            chosen = idx_bl if (idx_bl and (err_bl <= err_tl or not idx_tl)) else idx_tl
            if chosen and (k == 0 or len(chosen) == k):
                return chosen
    except Exception:
        return None
    return None


## 本地规则已移除；改由 LLM 方案生成


def _extract_json(text: str) -> dict:
    def _strip_comments(s: str) -> str:
        # 去掉行内 // 注释，便于解析不规范的 JSON 代码块
        return re.sub(r"//.*", "", s)

    def _try_load(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) 直接整体作为 JSON 解析
    obj = _try_load(text)
    if isinstance(obj, dict):
        return obj

    # 2) 优先从 ```json ... ``` 代码块中提取，允许带 // 注释
    m = re.search(r"```\s*json\s*\n([\s\S]*?)\n```", text, re.I)
    if m:
        inner = _strip_comments(m.group(1))
        obj = _try_load(inner)
        if isinstance(obj, dict):
            return obj

    # 3) 尝试从正文中抽取第一个 { ... }，同样容忍 // 注释
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        frag = _strip_comments(m.group(0))
        obj = _try_load(frag)
        if isinstance(obj, dict):
            return obj

    # 4) 最后兜底：提取关键信息（如“验证码为”），避免完全丢失结构化字段
    out: dict = {"summary": text.strip()[:500], "steps": []}
    try:
        m2 = re.search(r'["\']验证码为["\']\s*:\s*["\']([^"\']+)["\']', text)
        if m2:
            out["验证码为"] = m2.group(1).strip()
    except Exception:
        pass
    # 尝试从纯文本中提取坐标对，供 seq/grid/click 兜底使用
    try:
        pairs = re.findall(r"([0-9]{1,4})\s*,\s*([0-9]{1,4})", text)
        coords = []
        for a, b in pairs:
            try:
                coords.append({"x": int(a), "y": int(b)})
            except Exception:
                continue
        if coords:
            out.setdefault("标记坐标", coords)
            out.setdefault("执行坐标", coords)
    except Exception:
        pass
    return out


def _agent_guidance_direct(info: dict) -> Optional[dict]:
    """禁用直出，保持识别 Agent 自行推理。"""
    return None

    def _norm_value(val: str, norm: str) -> str:
        norm = (norm or "exact").lower()
        if norm == "lower":
            return (val or "").lower()
        if norm == "int":
            try:
                return str(int(float(val)))
            except Exception:
                return str(val)
        return str(val)

    input_kinds = {"text", "math", "case", "distort", "anagram", "sumdigits", "digitcount", "vowelcount", "hex2dec", "count"}
    if kind in input_kinds:
        norm = info.get("norm") or "exact"
        val = _norm_value(str(answer), norm)
        return {
            "summary": f"直出答案（{kind}）",
            "验证码为": val,
            "value": val,
            "steps": [
                {"label": "题目类型", "detail": f"{kind} 输入类"},
                {"label": "自检", "detail": "与存储答案一致"},
            ],
        }

    grid_kinds = {"grid", "gridcolor", "gridshape", "gridvowel"}
    seq_kinds = {"seq", "charseq", "arrowseq", "alphaseq", "numseq"}
    if kind in grid_kinds or kind in seq_kinds:
        try:
            parts = [p.strip() for p in str(answer).split(",") if p.strip()]
            idxs = [int(p) for p in parts]
        except Exception:
            idxs = []
        if not idxs:
            return None
        steps = [
            {"label": "题目类型", "detail": f"{kind} 九宫格"},
            {"label": "自检", "detail": "索引与存储答案一致"},
        ]
        return {
            "summary": f"直出索引（{kind}）",
            "indices": idxs,
            "steps": steps,
        }

    if kind in {"click", "odd"}:
        try:
            ax, ay_top = [float(v) for v in str(answer).split(",", 1)]
        except Exception:
            return None
        w, h = _ensure_image_size(info)
        if not h:
            try:
                h = int(float((info.get("meta") or {}).get("height") or 0))
            except Exception:
                h = 0
        ay_bl = int(round(h - ay_top)) if h else int(round(ay_top))
        point = {"x": int(round(ax)), "y": ay_bl}
        return {
            "summary": f"直出坐标（{kind}）",
            "point": dict(point),
            "in_target": True,
            "执行坐标": [dict(point)],
            "标记坐标": [dict(point)],
            "steps": [
                {"label": "题目类型", "detail": f"{kind} 点击类"},
                {"label": "自检", "detail": "坐标与存储答案一致"},
            ],
        }
    return None


def _agent_guidance_llm(info: dict) -> dict:
    # Construct CaptchaData
    # info contains merged entry (from store) and body (from client)
    # Try to recover SVG from data_uri if svg is missing
    svg_content = info.get("svg", "")
    data_uri = info.get("data_uri", "")
    if not svg_content and data_uri and "base64," in data_uri:
        try:
            import base64
            svg_content = base64.b64decode(data_uri.split("base64,")[1]).decode("utf-8")
        except Exception:
            pass

    captcha = CaptchaData(
        id=str(info.get("id", "")),
        kind=(info.get("kind") or info.get("type") or "").lower(),
        svg=svg_content,
        data_uri=data_uri,
        answer=str(info.get("answer", "")),
        expires_at=float(info.get("expires_at", 0)),
        norm=str(info.get("norm", "exact")),
        prompt=str(info.get("prompt", "")),
        meta=info.get("meta", {})
    )

    # Call Identification Agent
    # Pass image_png if available (as data URI) to avoid SVG compatibility issues with LLM
    image_png = info.get("image_png")
    if not (isinstance(image_png, str) and image_png.startswith("data:image/png")):
        image_png = None
        
    res = _ID_AGENT.solve(captcha, image_uri=image_png)

    # Convert AgentResult to expected JSON format
    ret = {
        "summary": res.reasoning or "Agent completed identification",
        "steps": [{"label": "Reasoning", "detail": res.reasoning}] if res.reasoning else [],
        "history": res.history
    }
    
    if res.answer:
        # For text/math types, the frontend/exec agent expects "验证码为" or just the value in summary/steps
        ret["验证码为"] = res.answer
        ret["value"] = res.answer # Helper for exec agent
        
    if res.click_point:
        ret["point"] = {"x": res.click_point[0], "y": res.click_point[1]}
        ret["标记坐标"] = [{"x": res.click_point[0], "y": res.click_point[1]}]
        # Also set "执行坐标" for consistency
        ret["执行坐标"] = [{"x": res.click_point[0], "y": res.click_point[1]}]
        
    if res.indices:
        ret["indices"] = res.indices
        
    return ret


def _agent_guidance_fallback(info: dict, err: Optional[str] = None) -> dict:
    kind = (info.get("kind") or info.get("type") or "").lower()
    norm = (info.get("norm") or "exact").lower()
    prompt = info.get("prompt") or "请完成验证"
    meta = info.get("meta") or {}
    w, h = _ensure_image_size(info)
    steps = []
    steps.append({"label": "题目类型", "detail": f"type={kind}；norm={norm}"})
    steps.append({"label": "题目要求", "detail": prompt})
    if w and h:
        steps.append({"label": "观察要点", "detail": f"图像尺寸 {w}×{h}"})
    # 通用指导（不暴露真值）
    steps.append({"label": "操作步骤", "detail": "请根据题面完成操作；如需坐标，请开启坐标工具并点击目标"})
    if err:
        steps.append({"label": "回退说明", "detail": f"LLM 不可用/超时：{err}"})
    return {"summary": f"本地回退：{kind} 指南（不含坐标/索引/真值）", "steps": steps}


def _agent_guidance(info: dict) -> dict:
    safe_info = _scrub_info(info)
    try:
        data = _agent_guidance_llm(safe_info)
    except Exception as e:
        data = _agent_guidance_fallback(safe_info, str(e))
    # click/odd：统一 point/标记坐标/执行坐标，优先使用执行坐标
    try:
        kind = (info.get("kind") or info.get("type") or "").lower()
        if kind in {"click", "odd"} and isinstance(data, dict):
            exec_pos = data.get("执行坐标")
            # 若执行坐标缺失，尝试用 point 或 标记坐标 填充
            if not exec_pos:
                pt = data.get("point")
                marks = data.get("标记坐标")
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    exec_pos = [pt]
                elif isinstance(marks, list) and marks:
                    exec_pos = [marks[0]]
            # 规范化执行坐标
            norm_exec = []
            if isinstance(exec_pos, dict):
                exec_pos = [exec_pos]
            if isinstance(exec_pos, list):
                for p in exec_pos:
                    if isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)):
                        norm_exec.append({"x": int(round(p["x"])), "y": int(round(p["y"]))})
            if norm_exec:
                data["执行坐标"] = norm_exec
                data["point"] = dict(norm_exec[0])
                data["标记坐标"] = list(norm_exec)
    except Exception:
        pass
    # 若复审标记 require_change=false，则不允许识别Agent更改点位，强制回退到 prev_point
    try:
        rev = info.get("revise") or {}
        prev = rev.get("prev_point")
        require_change = rev.get("require_change")
        if isinstance(prev, dict) and require_change is False:
            px = prev.get("x"); py = prev.get("y")
            if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                prev_point = {"x": int(round(px)), "y": int(round(py))}
                cur = data if isinstance(data, dict) else {}
                pt = (cur.get("point") or {})
                changed = not (isinstance(pt, dict) and int(round(pt.get("x", 0))) == prev_point["x"] and int(round(pt.get("y", 0))) == prev_point["y"])
                if cur.get("point") is None or changed:
                    cur["point"] = dict(prev_point)
                    cur["in_target"] = True
                # 同步标记/执行坐标
                cur["标记坐标"] = [dict(prev_point)]
                cur["执行坐标"] = [dict(prev_point)]
                steps = cur.get("steps") or cur.get("plan") or []
                steps.append({"label": "复审锁定", "detail": "require_change=false，保持上一轮坐标不变"})
                cur["steps"] = steps
                data = cur
    except Exception:
        pass
    return data


def _agent_exec_input_llm(info: dict, reco: dict) -> dict:
    # Reconstruct AgentResult from reco
    res = AgentResult(
        success=True,
        answer=reco.get("value") or reco.get("验证码为"),
        indices=reco.get("indices"),
        click_point=(reco.get("point", {}).get("x"), reco.get("point", {}).get("y")) if reco.get("point") else None
    )
    
    exec_res = _EXEC_AGENT.execute(res)
    
    # Map back to expected format
    return {
        "action": exec_res["action"],
        "value": exec_res.get("value", ""),
        "summary": reco.get("summary", "Execution completed"),
        "steps": reco.get("steps", [])
    }


def _agent_exec_click_llm(info: dict, reco: dict) -> dict:
    # Reconstruct AgentResult from reco
    res = AgentResult(
        success=True,
        answer=reco.get("value") or reco.get("验证码为"),
        indices=reco.get("indices"),
        click_point=(reco.get("point", {}).get("x"), reco.get("point", {}).get("y")) if reco.get("point") else None
    )
    
    exec_res = _EXEC_AGENT.execute(res)
    
    ret = {
        "action": exec_res["action"],
        "summary": reco.get("summary", "Execution completed"),
        "steps": reco.get("steps", [])
    }
    if exec_res["action"] == "click":
        ret["x"] = exec_res["x"]
        ret["y"] = exec_res["y"]
    elif exec_res["action"] == "input":
        # For grid/seq, if exec returns input (indices string), we might need to map it to indices list?
        # But the original _agent_exec_click_llm returned "grid" or "seq" action with indices.
        # My ExecutionAgent returns "input" with value string for grid/seq.
        # I should adapt ExecutionAgent or map it here.
        # If result.indices is present, I should return "grid" or "seq" action.
        if res.indices:
            kind = (info.get("kind") or info.get("type") or "").lower()
            if kind in ["seq", "charseq", "arrowseq", "alphaseq", "numseq"]:
                ret["action"] = "seq"
            else:
                ret["action"] = "grid"
            ret["indices"] = res.indices
        else:
            # Fallback for click if no point?
            pass
            
    return ret


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">

<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>验证码智能体演示</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --error: #ef4444;
            --radius: 12px;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        * {
            box-sizing: border-box;
        }

        .app-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
            height: 100vh;
            max-height: 100vh;
            overflow: hidden;
        }

        @media (max-width: 1024px) {
            .app-container {
                grid-template-columns: 1fr;
                height: auto;
                overflow: auto;
            }
        }

        .header {
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .brand {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .brand svg {
            width: 28px;
            height: 28px;
            color: var(--primary);
        }

        .card {
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .stage-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            padding: 1.5rem;
        }

        .stage-wrapper {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: #fff;
            align-self: center;
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.03);
            width: 100%;
            display: flex;
            justify-content: center;
            background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
            background-size: 20px 20px;
        }

        #cap-stage {
            position: relative;
            display: inline-block;
            max-width: 100%;
            min-width: 500px;
        }

        #cap-img,
        #cap-inline svg {
            display: block;
            width: 100%;
            height: auto;
            max-height: 80vh;
            object-fit: contain;
        }

        #cap-sel,
        #cap-axes {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .sel-box {
            position: absolute;
            border: 2px solid rgba(99, 102, 241, 0.9);
            background: rgba(99, 102, 241, 0.18);
            border-radius: 6px;
            box-sizing: border-box;
        }

        .controls {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid var(--border);
            background: #fff;
            color: var(--text-main);
            box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        }

        .btn:hover {
            background: #f1f5f9;
            border-color: #cbd5e1;
        }

        .btn-primary {
            background: var(--primary);
            color: #fff;
            border-color: transparent;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
        }

        .input-group {
            display: flex;
            gap: 0.5rem;
        }

        .input-field {
            padding: 0.6rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: inherit;
            width: 200px;
        }

        .type-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
            margin-top: auto;
        }

        .type-btn {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #fff;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
        }

        .type-btn:hover {
            color: var(--primary);
            border-color: var(--primary);
            background: #eef2ff;
        }

        .type-btn.active {
            background: var(--primary);
            color: #fff;
            border-color: transparent;
        }

        .agent-panel {
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            background: #f8fafc;
        }

        .panel-title {
            font-weight: 600;
            font-size: 1rem;
            color: var(--text-main);
        }

        .stepper {
            display: flex;
            justify-content: space-between;
            padding: 1.5rem;
            position: relative;
        }

        .stepper::before {
            content: '';
            position: absolute;
            top: 2.2rem;
            left: 2rem;
            right: 2rem;
            height: 2px;
            background: var(--border);
            z-index: 0;
        }

        .step {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }

        .step-circle {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #fff;
            border: 2px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-muted);
            transition: all 0.3s;
        }

        .step-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .step.active .step-circle {
            border-color: var(--primary);
            background: var(--primary);
            color: #fff;
            box-shadow: 0 0 0 4px #eef2ff;
        }

        .step.active .step-label {
            color: var(--primary);
            font-weight: 600;
        }

        .step.completed .step-circle {
            background: var(--success);
            border-color: var(--success);
            color: #fff;
        }

        .timeline-container {
            flex: 1;
            overflow-y: auto;
            padding: 0 1.5rem 1.5rem;
        }

        .timeline-item {
            position: relative;
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }

        .timeline-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0.4rem;
            width: 2px;
            height: 100%;
            background: var(--border);
        }

        .timeline-item:last-child::before {
            display: none;
        }

        .timeline-dot {
            position: absolute;
            left: -4px;
            top: 0.4rem;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            border: 2px solid #fff;
            box-shadow: 0 0 0 2px #eef2ff;
        }

        .timeline-content {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .timeline-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .timeline-body {
            font-size: 0.9rem;
            color: var(--text-muted);
            white-space: pre-wrap;
            word-break: break-word;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(5px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .json-viewer-container {
            margin: 0 1.5rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        .json-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.9rem 1rem 0.3rem;
            border-bottom: 1px solid rgba(99, 102, 241, 0.15);
        }

        .json-title {
            font-weight: 700;
            color: #0f172a;
            font-size: 0.95rem;
        }

        .json-subtitle {
            color: #475569;
            font-size: 0.8rem;
        }

        .json-actions {
            display: flex;
            gap: 0.4rem;
            align-items: center;
        }

        .pill-btn, .icon-btn {
            border: 1px solid #cbd5e1;
            background: #fff;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: #334155;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            transition: all 0.15s ease;
        }

        .pill-btn:hover, .icon-btn:hover {
            border-color: #a5b4fc;
            color: #4338ca;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.12);
            transform: translateY(-1px);
        }

        .icon-btn {
            padding: 0.35rem 0.55rem;
        }

        .json-viewer {
            background: #0f172a;
            color: #cbd5e1;
            padding: 1rem 1.25rem;
            font-family: 'Menlo', monospace;
            font-size: 0.82rem;
            overflow: auto;
            display: none;
            max-height: 320px;
            border-radius: 0 0 12px 12px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .json-viewer.open {
            display: block;
        }

        #toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: var(--text-main);
            color: #fff;
            border-radius: 8px;
            box-shadow: var(--shadow);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
            z-index: 100;
        }

        #toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .hidden {
            display: none !important;
        }
    </style>
</head>

<body>
    <div class="app-container">
        <div class="header">
            <div class="brand">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                <span>CAPTCHA Agent</span>
            </div>
            <div class="controls">
                <button class="btn" onclick="loadCaptcha(currentType || 'text')">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    刷新题目
                </button>
            </div>
        </div>

        <div class="card stage-panel">
            <div class="stage-wrapper">
                <div id="cap-stage">
                    <img id="cap-img" src="" alt="captcha" style="display:none" />
                    <div id="cap-inline"></div>
                    <canvas id="cap-axes"></canvas>
                    <div id="cap-sel"></div>
                </div>
            </div>

            <div class="controls">
                <div class="input-group" id="manual-input-group">
                    <input id="cap-input" class="input-field" placeholder="输入验证码" />
                    <button class="btn" onclick="verify()">验证</button>
                </div>
                <button id="btn-agent" class="btn btn-primary" onclick="runAgent()">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    运行智能体
                </button>
                <label class="btn" style="font-size:0.85rem; padding:0.4rem 0.8rem;">
                    <input type="checkbox" id="toggle-gridlines" checked onchange="updateHighlights()"> 网格
                </label>
                <label class="btn" style="font-size:0.85rem; padding:0.4rem 0.8rem;">
                    <input type="checkbox" id="toggle-axes" onchange="updateAxes()"> 坐标轴
                </label>
            </div>

            <div class="type-selector" id="type-buttons"></div>

            <div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 1rem;">
                <span id="cap-desc">类型: Text</span>
                <div id="cap-click-tip"
                    style="color: var(--primary); font-weight: 500; margin-top: 0.5rem; display: none;">点击图片选择目标</div>
            </div>
        </div>

        <div class="card agent-panel">
            <div class="panel-header">
                <div class="panel-title">智能体思维链</div>
            </div>

            <div class="stepper">
                <div class="step" id="step-1">
                    <div class="step-circle">1</div>
                    <div class="step-label">识别</div>
                </div>
                <div class="step" id="step-2">
                    <div class="step-circle">2</div>
                    <div class="step-label">复审</div>
                </div>
                <div class="step" id="step-3">
                    <div class="step-circle">3</div>
                    <div class="step-label">执行</div>
                </div>
                <div class="step" id="step-4">
                    <div class="step-circle">4</div>
                    <div class="step-label">验证</div>
                </div>
            </div>

            <div class="timeline-container" id="timeline">
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <div class="timeline-header">系统就绪</div>
                        <div class="timeline-body">等待任务开始...</div>
                    </div>
                </div>
            </div>

            <div class="json-viewer-container">
                <div class="json-header">
                    <div>
                        <div class="json-title">原始 JSON</div>
                        <div class="json-subtitle">查看每一步的原始数据流</div>
                    </div>
                    <div class="json-actions">
                        <button id="json-toggle-btn" class="pill-btn" onclick="toggleJson()">
                            展开
                        </button>
                        <button class="icon-btn" onclick="copyJson()" title="复制 JSON">
                            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                            </svg>
                        </button>
                    </div>
                </div>
                <pre id="json-viewer" class="json-viewer"></pre>
            </div>
        </div>
    </div>

    <div id="toast"></div>

    <script>
        let currentId = null;
        let currentType = null;
        let clickAnswer = null;
        let agentPoint = null;
        let currentMeta = null;
        let gridSelected = new Set();
        let seqSelected = [];
        let axesGridN = 10;
        let lastPayload = null;
        let agentResult = null;
        let executedOnce = false;
        let processLog = {}; // Store complete process data

        const typeNames = {
            text: '字符型', math: '算术型', distort: '扭曲字符', subset: '指定位置', click: '点选图形', case: '区分大小写', sumdigits: '数字求和', count: '计数图形', grid: '网格多选', seq: '顺序点击', charseq: '字符顺序', odd: '找不同',
            gridvowel: '选择元音', arrowseq: '箭头顺序', anagram: '字母复原', compare: '比较大小', digitcount: '数字个数', vowelcount: '元音个数', palin: '回文判断', hex2dec: '十六进制转十进制',
            gridcolor: '按颜色多选', gridshape: '按形状多选', alphaseq: '字母顺序', numseq: '数字顺序'
        };
        const CATS = [
            { key: '输入类', types: ['text', 'math', 'case', 'distort', 'anagram'] },
            { key: '点选/九宫格', types: ['click', 'odd', 'grid', 'gridcolor', 'gridshape'] },
            { key: '顺序点击', types: ['seq', 'arrowseq', 'charseq', 'alphaseq', 'numseq'] },
            { key: '统计/规则', types: ['sumdigits', 'digitcount', 'vowelcount', 'hex2dec', 'count'] }
        ];

        function updateStepper(step) {
            for (let i = 1; i <= 4; i++) {
                const el = document.getElementById('step-' + i);
                el.className = 'step';
                if (i < step) el.classList.add('completed');
                if (i === step) el.classList.add('active');
            }
        }

        function addTimelineItem(title, body) {
            const c = document.getElementById('timeline');
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `<div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-header">${title}</div><div class="timeline-body">${body}</div></div>`;
            c.appendChild(item);
            c.scrollTop = c.scrollHeight;
        }

        function clearTimeline() {
            document.getElementById('timeline').innerHTML = '';
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), 3000);
        }

        function toggleJson() {
            document.getElementById('json-viewer').classList.toggle('open');
            const t = document.getElementById('json-toggle-btn');
            if (t) {
                t.textContent = document.getElementById('json-viewer').classList.contains('open') ? '收起' : '展开';
            }
        }

        function logJSON(label, obj) {
            // Accumulate process data
            processLog[label] = obj;
            const v = document.getElementById('json-viewer');
            v.textContent = JSON.stringify(processLog, null, 2);
        }

        function copyJson() {
            const v = document.getElementById('json-viewer');
            if (!v.textContent) {
                showToast('没有可复制的内容');
                return;
            }
            navigator.clipboard.writeText(v.textContent).then(() => {
                showToast('已复制到剪贴板');
            }).catch(err => {
                showToast('复制失败: ' + err);
            });
        }

        function renderTypeButtons() {
            const wrap = document.getElementById('type-buttons');
            wrap.innerHTML = '';
            CATS.forEach(cat => {
                cat.types.forEach(t => {
                    const b = document.createElement('button');
                    b.className = 'type-btn';
                    b.textContent = typeNames[t] || t;
                    b.onclick = () => {
                        document.querySelectorAll('.type-btn').forEach(btn => btn.classList.remove('active'));
                        b.classList.add('active');
                        loadCaptcha(t);
                    };
                    wrap.appendChild(b);
                });
            });
        }

        async function loadCaptcha(kind) {
            updateStepper(0);
            clearTimeline();
            addTimelineItem('系统就绪', '题目已刷新');
            const r = await fetch(`/captcha?type=${encodeURIComponent(kind)}`);
            const j = await r.json();
            currentId = j.id;
            currentType = j.type;
            lastPayload = j;

            const inline = document.getElementById('cap-inline');
            inline.innerHTML = j.svg || '';
            document.getElementById('cap-desc').textContent = `类型: ${typeNames[j.type] || j.type} | ID: ${j.id}`;

            setUIForType(j.type, j.meta || {});
            updateHighlights();
            
            // Reset process log for new captcha
            processLog = {};
            logJSON('1_题目数据', j);
        }

        function setUIForType(t, meta) {
            const inputGroup = document.getElementById('manual-input-group');
            const inputField = document.getElementById('cap-input');
            const clickTip = document.getElementById('cap-click-tip');
            const gridToggleLabel = document.querySelector('label:has(#toggle-gridlines)');
            const axesToggleLabel = document.querySelector('label:has(#toggle-axes)');
            const gridToggle = document.getElementById('toggle-gridlines');
            const axesToggle = document.getElementById('toggle-axes');

            // Always clear previous manual input when refreshing/changing type
            if (inputField) inputField.value = '';
            
            // Classify CAPTCHA types
            const textTypes = ['text', 'math', 'case', 'distort', 'anagram', 'sumdigits', 'digitcount', 'vowelcount', 'hex2dec', 'count'];
            const pointSelectTypes = ['click', 'odd'];
            const gridTypes = ['grid', 'gridcolor', 'gridshape', 'gridvowel', 'seq', 'charseq', 'arrowseq', 'alphaseq', 'numseq'];
            
            const isText = textTypes.includes(t);
            const isPointSelect = pointSelectTypes.includes(t);
            const isGrid = gridTypes.includes(t);
            const isClick = isPointSelect || isGrid;

            inputGroup.style.display = isClick ? 'none' : 'flex';
            clickTip.style.display = isClick ? 'block' : 'none';
            
            // Conditional toggle visibility AND state reset
            if (isText) {
                // Text-based: hide both toggles and uncheck them
                if (gridToggleLabel) gridToggleLabel.style.display = 'none';
                if (axesToggleLabel) axesToggleLabel.style.display = 'none';
                if (gridToggle) gridToggle.checked = false;
                if (axesToggle) axesToggle.checked = false;
            } else if (isPointSelect) {
                // Point-select: show axes toggle only, uncheck grid
                if (gridToggleLabel) gridToggleLabel.style.display = 'none';
                if (axesToggleLabel) axesToggleLabel.style.display = 'inline-flex';
                if (gridToggle) gridToggle.checked = false;
                // Don't auto-check axes - keep it as a tool
            } else if (isGrid) {
                // Grid-based: show grid toggle only, uncheck axes
                if (gridToggleLabel) gridToggleLabel.style.display = 'inline-flex';
                if (axesToggleLabel) axesToggleLabel.style.display = 'none';
                if (axesToggle) axesToggle.checked = false;
                // Keep grid checked by default for grid types
                if (gridToggle) gridToggle.checked = true;
            }

            currentMeta = meta;
            gridSelected = new Set();
            seqSelected = [];
            clickAnswer = null;
            agentPoint = null;
            
            // Update UI elements
            updateHighlights();
            updateAxes();
        }

        function getStageSize() {
            const img = document.getElementById('cap-img');
            const stage = document.getElementById('cap-stage');
            const rect = stage.getBoundingClientRect();
            let natW = img && img.naturalWidth ? img.naturalWidth : 0;
            let natH = img && img.naturalHeight ? img.naturalHeight : 0;
            if ((!natW || !natH) && lastPayload && lastPayload.svg) {
                try {
                    const m1 = lastPayload.svg.match(/viewBox\s*=\s*['\"]\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*['\"]/i);
                    if (m1) { natW = parseFloat(m1[1]); natH = parseFloat(m1[2]); }
                } catch (e) { }
            }
            return { displayW: rect.width, displayH: rect.height, naturalW: natW || rect.width, naturalH: natH || rect.height, rect };
        }

        function updateAxes() {
            const axes = document.getElementById('cap-axes');
            const show = document.getElementById('toggle-axes');
            const sz = getStageSize();

            // High-DPI support
            const dpr = window.devicePixelRatio || 1;
            const w = Math.floor(sz.rect.width);
            const h = Math.floor(sz.rect.height);

            axes.style.width = w + 'px';
            axes.style.height = h + 'px';
            axes.width = w * dpr;
            axes.height = h * dpr;

            const ctx = axes.getContext('2d');
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, w, h);

            if (!show || !show.checked) return;

            // Draw coordinate system for point-select types
            const pointSelectTypes = ['click', 'odd'];
            const isPointSelect = pointSelectTypes.includes(currentType);
            
            if (isPointSelect) {
                // Enhanced coordinate axis with labels
                const natW = sz.naturalW;
                const natH = sz.naturalH;
                
                // Draw main axes (Bottom and Left)
                ctx.strokeStyle = 'rgba(0,0,0,0.8)';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(0, h); ctx.lineTo(w, h); // X-axis (bottom)
                ctx.moveTo(0, 0); ctx.lineTo(0, h);  // Y-axis (left)
                ctx.stroke();
                
                // Draw origin label
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                ctx.font = 'bold 12px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'bottom';
                ctx.fillText('(0,0)', 4, h - 4);
                
                // Draw gridlines and labels (every 10%)
                ctx.strokeStyle = 'rgba(99,102,241,0.5)';
                ctx.lineWidth = 2;
                ctx.font = '10px Inter, sans-serif';
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                
                // Vertical gridlines (X-axis intervals)
                const xIntervals = 10;
                for (let i = 1; i < xIntervals; i++) {
                    const x = Math.round(w * i / xIntervals) + 0.5;
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
                    
                    // X-axis labels (bottom)
                    const xVal = Math.round(natW * i / xIntervals);
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillText(String(xVal), x, h - 16);
                }
                
                // Horizontal gridlines (Y-axis intervals)
                const yIntervals = 10;
                for (let i = 1; i < yIntervals; i++) {
                    const y = Math.round(h * i / yIntervals) + 0.5;
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
                    
                    // Y-axis labels (left) - remember Y is inverted (0 at bottom)
                    const yVal = Math.round(natH * (yIntervals - i) / yIntervals);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(yVal), 4, y);
                }
                
                // Draw max value labels
                ctx.font = 'bold 11px Inter, sans-serif';
                ctx.fillStyle = 'rgba(0,0,0,0.8)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(String(natW), w - 10, h - 16);
                
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(String(natH), 4, 4);
            } else {
                // Simple gridlines for non-coordinate types
                ctx.strokeStyle = 'rgba(99,102,241,0.3)';
                ctx.lineWidth = 1;
                const N = 10;
                for (let i = 1; i < N; i++) {
                    const x = Math.round(w * i / N) + 0.5;
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
                    const y = Math.round(h * i / N) + 0.5;
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
                }
            }

            if (clickAnswer) {
                const scaleX = sz.naturalW ? (w / sz.naturalW) : 1;
                const scaleY = sz.naturalH ? (h / sz.naturalH) : 1;
                const cx = clickAnswer.x * scaleX;
                const cy = h - (clickAnswer.y * scaleY);
                ctx.fillStyle = '#6366f1';
                ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
            }
            if (agentPoint) {
                const scaleX = sz.naturalW ? (w / sz.naturalW) : 1;
                const scaleY = sz.naturalH ? (h / sz.naturalH) : 1;
                const cx = agentPoint.x * scaleX;
                const cy = h - (agentPoint.y * scaleY);
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8); ctx.stroke();
            }
        }

        function updateHighlights() {
            const meta = currentMeta || {};
            const rows = meta.rows || 3;
            const cols = meta.cols || 3;
            const sz = getStageSize();
            const rect = sz.rect;
            const cont = document.getElementById('cap-sel');
            cont.innerHTML = '';

            const gridToggle = document.getElementById('toggle-gridlines');
            const gridTypes = ['grid', 'gridcolor', 'gridshape', 'seq', 'charseq', 'arrowseq', 'alphaseq', 'numseq'];
            if (gridToggle && gridToggle.checked && gridTypes.includes(currentType)) {
                const cellW = rect.width / cols, cellH = rect.height / rows;
                cont.style.backgroundImage = `linear-gradient(to right, rgba(0,0,0,0.15) 2px, transparent 2px), linear-gradient(to bottom, rgba(0,0,0,0.15) 2px, transparent 2px)`;
                cont.style.backgroundSize = `${cellW}px ${cellH}px, ${cellW}px ${cellH}px`;
                
                // Draw grid row/column labels
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const idx = r * cols + c;
                        const x = c * cellW;
                        const y = r * cellH;
                        
                        // Create label element
                        const label = document.createElement('div');
                        label.style.position = 'absolute';
                        label.style.left = x + 'px';
                        label.style.top = y + 'px';
                        label.style.width = cellW + 'px';
                        label.style.height = cellH + 'px';
                        label.style.display = 'flex';
                        label.style.alignItems = 'flex-start';
                        label.style.justifyContent = 'flex-start';
                        label.style.padding = '2px';
                        label.style.fontSize = '10px';
                        label.style.fontWeight = 'bold';
                        label.style.color = 'rgba(99,102,241,0.6)';
                        label.style.pointerEvents = 'none';
                        label.textContent = String(idx);
                        cont.appendChild(label);
                    }
                }
            } else {
                cont.style.backgroundImage = 'none';
            }

            const addBox = (idx, label) => {
                const r = Math.floor(idx / cols), c = idx % cols;
                const div = document.createElement('div');
                div.className = 'sel-box';
                div.style.left = (c * rect.width / cols) + 'px';
                div.style.top = (r * rect.height / rows) + 'px';
                div.style.width = (rect.width / cols) + 'px';
                div.style.height = (rect.height / rows) + 'px';
                if (label) { div.textContent = label; div.style.color = '#fff'; div.style.fontWeight = 'bold'; div.style.display = 'flex'; div.style.alignItems = 'center'; div.style.justifyContent = 'center'; div.style.fontSize = '1.2rem'; }
                cont.appendChild(div);
            };

            gridSelected.forEach(i => addBox(i));
            seqSelected.forEach((i, idx) => addBox(i, String(idx + 1)));
            updateAxes();
        }

        function onImageClick(ev) {
            const stage = document.getElementById('cap-stage');
            const sz = getStageSize();
            const rect = sz.rect;
            const x = ev.clientX - rect.left;
            const y = ev.clientY - rect.top;

            if (['grid', 'gridcolor', 'gridshape', 'seq', 'charseq', 'numseq'].includes(currentType)) {
                const meta = currentMeta || {};
                const rows = meta.rows || 3;
                const cols = meta.cols || 3;
                const col = Math.max(0, Math.min(cols - 1, Math.floor((x / rect.width) * cols)));
                const row = Math.max(0, Math.min(rows - 1, Math.floor((y / rect.height) * rows)));
                const idx = row * cols + col;

                if (currentType.startsWith('grid')) {
                    if (gridSelected.has(idx)) gridSelected.delete(idx); else gridSelected.add(idx);
                } else {
                    const pos = seqSelected.indexOf(idx);
                    if (pos >= 0) seqSelected = seqSelected.filter(v => v !== idx); else seqSelected.push(idx);
                }
                updateHighlights();
                return;
            }

            if (currentType === 'click' || currentType === 'odd') {
                const scaleX = sz.naturalW / rect.width;
                const scaleY = sz.naturalH / rect.height;
                const xp = Math.round(x * scaleX);
                const yp_top = Math.round(y * scaleY);
                const y_bl = Math.max(0, Math.min(sz.naturalH, Math.round(sz.naturalH - yp_top)));
                clickAnswer = { x: xp, y: y_bl };
                updateAxes();
            }
        }

        async function runAgent() {
            if (!currentId) return;
            updateStepper(1);
            clearTimeline();
            addTimelineItem('开始识别', 'Agent 正在分析图像...');

            const body = { id: currentId, ...lastPayload };

            // Generate PNG for agent (simplified)
            try {
                const sz = getStageSize();
                // Use natural size for agent, but ensure it's not 0
                const w = Math.max(1, Math.floor(sz.naturalW || 0));
                const h = Math.max(1, Math.floor(sz.naturalH || 0));
                const canvas = document.createElement('canvas');
                canvas.width = w; canvas.height = h;
                const ctx = canvas.getContext('2d');
                const im = new Image();
                await new Promise((resolve) => { im.onload = () => { try { ctx.drawImage(im, 0, 0, w, h); } catch (e) { } resolve(); }; im.onerror = () => resolve(); im.src = (lastPayload && lastPayload.data_uri) || ''; });

                // Draw grid if needed
                const t = lastPayload.type;
                if (['grid', 'gridcolor', 'gridshape', 'seq', 'charseq', 'numseq'].includes(t)) {
                    const rows = (lastPayload.meta && lastPayload.meta.rows) || 3;
                    const cols = (lastPayload.meta && lastPayload.meta.cols) || 3;
                    ctx.save();
                    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
                    ctx.lineWidth = 2;
                    for (let i = 0; i <= cols; i++) { const x = Math.round(w * i / cols); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
                    for (let i = 0; i <= rows; i++) { const y = Math.round(h * i / rows); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
                    ctx.restore();
                }

                body.image_png = canvas.toDataURL('image/png');
                body.image_width = w;
                body.image_height = h;
            } catch (e) { }

            const r = await fetch('/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const j = await r.json();
            agentResult = j;
            logJSON('2_识别结果', j);

            if (j.error) {
                addTimelineItem('识别失败', j.error);
                return;
            }

            (j.steps || []).forEach(s => addTimelineItem(s.label || '思考', s.detail));

            // Review logic (Simplified: 1 round)
            updateStepper(2);
            addTimelineItem('复审完成', '准备执行...');

            // Exec logic
            updateStepper(3);
            const execBody = { id: currentId, intent: ['click', 'odd', 'grid', 'seq'].includes(currentType) ? 'click' : 'input', reco: agentResult };
            const r2 = await fetch('/exec', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(execBody) });
            const j2 = await r2.json();
            logJSON('3_执行结果', j2);

            if (j2.action === 'click') {
                // click point
                if (j2.x !== undefined && j2.y !== undefined) {
                    agentPoint = { x: j2.x, y: j2.y }; // For visualization
                    updateAxes();
                }
            } else if (j2.action === 'grid' || j2.action === 'seq') {
                if (j2.indices) {
                    gridSelected = new Set(j2.indices);
                    if (j2.action === 'seq') seqSelected = j2.indices;
                    updateHighlights();
                }
            } else {
                document.getElementById('cap-input').value = j2.value;
            }

            // Verify logic
            updateStepper(4);
            await verify(true);
        }

        async function verify(silent = false) {
            let answer = '';
            if (['click', 'odd'].includes(currentType)) {
                if (!clickAnswer && !agentPoint) { if (!silent) showToast('请先选择目标'); return; }
                // If manual click exists, use it; otherwise use agent point (for auto run)
                const pt = clickAnswer || agentPoint;
                answer = pt.x + ',' + pt.y;
            } else if (currentType.startsWith('grid')) {
                answer = Array.from(gridSelected).sort((a, b) => a - b).join(',');
            } else if (currentType.endsWith('seq')) {
                answer = seqSelected.join(',');
            } else {
                answer = document.getElementById('cap-input').value;
            }

            const r = await fetch('/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: currentId, answer, origin: (currentType === 'click' || currentType === 'odd') ? 'bl' : undefined }) });
            const j = await r.json();
            
            // Log verification result
            logJSON('4_验证结果', j);

            if (j.ok) {
                updateStepper(4);
                document.getElementById('step-4').classList.add('completed');
                addTimelineItem('验证成功', '✅ 答案正确');
                showToast('验证通过');
            } else {
                addTimelineItem('验证失败', '❌ ' + j.message);
                showToast('验证失败');
            }
        }

        window.addEventListener('load', () => {
            document.getElementById('cap-stage').addEventListener('click', onImageClick);
            window.addEventListener('resize', updateHighlights);
            renderTypeButtons();
            loadCaptcha('text');
        });
    </script>
</body>

</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CaptchaServer/1.0"

    def _set_common_headers(self, code: int = 200, content_type: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._set_common_headers(200, "text/html; charset=utf-8")
            html = INDEX_HTML
            # 修正历史遗留的反斜杠转义，确保属性如 id="cap-img" 被浏览器正确解析
            # 反复替换直至没有残留
            while '\\"' in html:
                html = html.replace('\\"', '"')
            self.wfile.write(html.encode("utf-8"))
            return

        if parsed.path.startswith("/img/") and parsed.path.endswith(".png"):
            img_id = parsed.path[len("/img/"):-len(".png")]
            entry = _STORE.get(img_id)
            if not entry:
                self._set_common_headers(404)
                self.wfile.write(_json_bytes({"error": "not_found"}))
                return
            data_uri = entry.get("image_png")
            if not data_uri or not isinstance(data_uri, str) or "," not in data_uri:
                self._set_common_headers(404)
                self.wfile.write(_json_bytes({"error": "no_image"}))
                return
            try:
                import base64
                b64 = data_uri.split(",", 1)[1]
                content = base64.b64decode(b64)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": "decode_failed"}))
                return

        if parsed.path == "/captcha":
            params = parse_qs(parsed.query)
            kind = (params.get("type", ["text"]) or ["text"])[0]
            gen = _FACTORY.get(kind)
            c = gen.generate(ttl_seconds=_TTL_SECONDS)
            # 如果 TTL 关闭，强制设置为永不过期
            if _TTL_SECONDS <= 0:
                c.expires_at = float("inf")
            _STORE.put(
                c.id,
                c.answer,
                c.expires_at,
                c.norm,
                kind=getattr(c, "kind", None),
                prompt=getattr(c, "prompt", None),
                meta=getattr(c, "meta", None),
            )

            payload = {
                "id": c.id,
                "type": c.kind,
                "expires_in": None if _TTL_SECONDS <= 0 else _TTL_SECONDS,
                "svg": c.svg,
                "data_uri": c.data_uri,
                "norm": c.norm,
            }
            if getattr(c, "prompt", None):
                payload["prompt"] = c.prompt
            if getattr(c, "meta", None):
                payload["meta"] = c.meta
            if _DEBUG:
                payload["debug_answer"] = c.answer

            self._set_common_headers(200)
            self.wfile.write(_json_bytes(payload))
            return

        self._set_common_headers(404)
        self.wfile.write(_json_bytes({"error": "not_found"}))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/verify":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            answer = str(body.get("answer", ""))
            # 兼容外部传入“左下角为原点”的 point 坐标：origin=bl
            entry = _STORE.get(id_) or {}
            origin = str(body.get("origin", "")).lower()
            if origin == 'bl':
                try:
                    norm = (entry.get("norm") or "").lower()
                    if norm.startswith("point:"):
                        # 解析 y_bl，并转换为 y_top 以与存储答案保持一致
                        ax, ay = [float(v) for v in answer.split(",", 1)]
                        h = 0
                        # 优先使用显式传入的 image_height；否则尝试 meta.height
                        try:
                            h = int(float(body.get("image_height") or 0))
                        except Exception:
                            h = 0
                        if not h:
                            try:
                                meta = entry.get("meta") or {}
                                h = int(float(meta.get("height") or 0))
                            except Exception:
                                h = 0
                        if h:
                            ay_top = h - ay
                            answer = f"{int(round(ax))},{int(round(ay_top))}"
                except Exception:
                    pass
            ok, msg = _STORE.verify(id_, answer)
            if not ok:
                entry = _STORE.get(id_)
                if entry:
                    print(f"VERIFY FAIL: ID={id_}, Expected={entry.get('answer')}, Got={answer}, Norm={entry.get('norm')}")
            self._set_common_headers(200)
            self.wfile.write(_json_bytes({"ok": ok, "message": msg}))
            return
        if parsed.path == "/agent":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            info = _STORE.get(id_) or {}
            # 如果 GET /captcha 的响应未包含某些字段，尝试补齐
            for k in ("type","kind","prompt","meta","norm","data_uri","image_png","image_width","image_height"):
                if k in body and k not in info:
                    info[k] = body[k]
            if "revise" in body and "revise" not in info:
                info["revise"] = body["revise"]
            # 如果前端传了 image_png，则保存，并生成可被外网访问的 URL（基于 Host）
            if body.get("image_png"):
                _STORE.update(id_, image_png=body.get("image_png"))
                host = (self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "localhost").lower()
                proto = (self.headers.get("X-Forwarded-Proto") or ("https" if self.server.server_address[1] == 443 else "http")).lower()
                # 仅当 Host 不是本机地址时，才生成对外 URL
                if not (host.startswith("127.") or host.startswith("localhost") or host.startswith("0.0.0.0")):
                    info["image_url"] = f"{proto}://{host}/img/{id_}.png"
            try:
                data = _agent_guidance(_scrub_info(info))
                self._set_common_headers(200)
                self.wfile.write(_json_bytes(data))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": str(e)}))
            return
        if parsed.path == "/exec":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            intent = str(body.get("intent", "")).strip().lower()
            reco = body.get("reco") or {}
            info = _STORE.get(id_) or {}
            for k in ("type","kind","prompt","meta","norm","data_uri","image_png","image_width","image_height"):
                if k in body and k not in info:
                    info[k] = body[k]
            try:
                if intent == "input":
                    data = _agent_exec_input_llm(_scrub_info(info), reco)
                elif intent == "click":
                    data = _agent_exec_click_llm(_scrub_info(info), reco)
                else:
                    raise RuntimeError("invalid_intent")
                self._set_common_headers(200)
                self.wfile.write(_json_bytes(data))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": str(e)}))
            return
        

        self._set_common_headers(404)
        self.wfile.write(_json_bytes({"error": "not_found"}))


def _cleanup_task():
    while True:
        try:
            if _TTL_SECONDS <= 0:
                time.sleep(300)
                continue
            _STORE.cleanup()
        except Exception:
            pass
        time.sleep(30)


def run_server(host: str = "127.0.0.1", port: int = 8000, ttl_seconds: int = 120, debug: bool = False):
    global _TTL_SECONDS, _DEBUG
    try:
        _TTL_SECONDS = int(ttl_seconds)
    except Exception:
        _TTL_SECONDS = 0
    if _TTL_SECONDS > 0:
        _TTL_SECONDS = max(10, _TTL_SECONDS)
    _DEBUG = bool(debug)

    th = threading.Thread(target=_cleanup_task, daemon=True)
    th.start()

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port} (ttl={_TTL_SECONDS}s, debug={_DEBUG}) …")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
