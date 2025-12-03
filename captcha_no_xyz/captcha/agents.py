import os
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # Optional dependency for LLM solve flow
    OpenAI = None  # type: ignore

from .generators import CaptchaData
try:
    from pydantic import BaseModel
except ModuleNotFoundError:  # pydantic is used only for type hints here
    class BaseModel:  # type: ignore
        pass

class TextAnswer(BaseModel):
    answer: str
    reasoning: str

class ClickAnswer(BaseModel):
    x: int
    y: int
    reasoning: str

class GridAnswer(BaseModel):
    indices: List[int]
    reasoning: str

@dataclass
class AgentResult:
    success: bool
    answer: Optional[str] = None
    click_point: Optional[Tuple[int, int]] = None
    indices: Optional[List[int]] = None
    reasoning: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

class AgentBase:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.qingyuntop.top/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")
        self.client = None
        self._init_error = None

        if not OpenAI:
            self._init_error = "openai package not installed; install `openai` to enable LLM solving"
        elif not self.api_key:
            self._init_error = "OPENAI_API_KEY not set; LLM solving is disabled"
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call_llm(self, messages: List[Dict[str, Any]], response_format=None) -> Dict[str, Any]:
        if not self.client:
            return {"error": self._init_error or "OpenAI client not initialized"}
        
        try:
            # Fallback for standard JSON mode or text
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=60,
            )
            content = completion.choices[0].message.content
            if not content:
                return {"error": "Empty response from LLM"}
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the content if it's wrapped in markdown
                import re
                # 1. Try markdown code block
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        return json.loads(json_str)
                    except:
                        pass
                
                # 2. Try regex to find the first JSON object
                try:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                except:
                    pass
                    
                return {"error": f"Invalid JSON response: {content[:100]}..."}
        except Exception as e:
            return {"error": str(e)}

    def _verify_local(self, captcha: CaptchaData, user_answer: str) -> bool:
        """Replicates store.verify logic without consuming the captcha."""
        answer = captcha.answer
        norm = captcha.norm
        ua_raw = (user_answer or "").strip()
        
        if norm == "exact":
            return ua_raw == answer
        elif norm == "lower":
            return ua_raw.lower() == answer.lower()
        elif norm == "int":
            try:
                return int(ua_raw) == int(answer)
            except Exception:
                return False
        elif norm.startswith("point:"):
            try:
                tol = float(norm.split(":", 1)[1])
                ax, ay = [float(v) for v in answer.split(",", 1)]
                ux, uy = [float(v) for v in ua_raw.split(",", 1)]
                dx = ux - ax
                dy = uy - ay
                ok = (dx * dx + dy * dy) ** 0.5 <= tol
                if ok:
                    return True
                # tolerate tl/bl mismatch using meta.height
                try:
                    h = float((captcha.meta or {}).get("height") or 0)
                except Exception:
                    h = 0
                if h:
                    candidates = [
                        (ux, h - uy, ax, ay),
                        (ux, uy, ax, h - ay),
                        (ux, h - uy, ax, h - ay),
                    ]
                    for uxo, uyo, axo, ayo in candidates:
                        dx = uxo - axo
                        dy = uyo - ayo
                        if (dx * dx + dy * dy) ** 0.5 <= tol:
                            return True
                return False
            except Exception:
                return False
        elif norm.startswith("approx:"):
            try:
                tol = int(float(norm.split(":", 1)[1]))
                return abs(int(ua_raw) - int(answer)) <= tol
            except Exception:
                return False
        elif norm == "set":
            try:
                a_set = set(int(x) for x in filter(None, (s.strip() for s in answer.split(','))))
                u_set = set(int(x) for x in filter(None, (s.strip() for s in ua_raw.split(','))))
                return a_set == u_set
            except Exception:
                return False
        elif norm == "list":
            try:
                a_list = [int(x) for x in filter(None, (s.strip() for s in answer.split(',')))]
                u_list = [int(x) for x in filter(None, (s.strip() for s in ua_raw.split(',')))]
                return a_list == u_list
            except Exception:
                return False
        else:
            return ua_raw == answer

class IdentificationAgent(AgentBase):
    def _consistent_with_prev(self, history: list[Dict[str, Any]], current: str) -> bool:
        """Check whether current answer matches previous attempt."""
        if not history:
            return False
        prev = history[-1].get("answer")
        return prev is not None and str(prev) == str(current)

    def _anagram_word_ok(self, captcha: CaptchaData, ans: str) -> bool:
        """Agent-side复审：anagram 结果必须是一个英文单词；若有候选列表则必须在列表内。"""
        if not isinstance(ans, str):
            return False
        word = ans.strip()
        if not word.isalpha():
            return False
        if len(word) < 3:
            return False
        try:
            meta_words = (captcha.meta or {}).get("words") or []
            allowed = {str(w).lower() for w in meta_words if isinstance(w, str)}
            if allowed:
                return word.lower() in allowed
        except Exception:
            pass
        return True

    def _point_in_target(self, x: int, y: int, target: Optional[Dict[str, Any]]) -> bool:
        if not target:
            return False
        try:
            cx = float(target.get("cx", 0))
            cy = float(target.get("cy", 0))
            tol = float(target.get("tol", 0))
            dx = float(x) - cx
            dy = float(y) - cy
            return (dx * dx + dy * dy) ** 0.5 <= max(tol, 1.0)
        except Exception:
            return False

    def _resolve_click_target(self, captcha: CaptchaData) -> Optional[Dict[str, Any]]:
        """Infer target region for click/odd using meta + prompt, without暴露答案."""
        meta = captcha.meta or {}
        prompt = captcha.prompt or ""
        # 捕获颜色/形状关键词
        kinds = ["圆形", "方形", "三角形", "五角星"]
        target_kind = next((k for k in kinds if k in prompt), None)
        palette = meta.get("palette") or []
        color_names = [it.get("name") if isinstance(it, dict) else (it[0] if isinstance(it, (list, tuple)) else None) for it in palette]
        target_color = next((n for n in color_names if isinstance(n, str) and n in prompt), None)

        # items 可能是 tuple 也可能是 dict
        items = meta.get("items") or []
        candidates = []
        for it in items:
            if isinstance(it, dict):
                k = it.get("kind"); c = it.get("color"); cx = it.get("cx"); cy = it.get("cy"); size = it.get("size")
            elif isinstance(it, (list, tuple)) and len(it) >= 6:
                k, c, _, cx, cy, size = it[:6]
            else:
                continue
            candidates.append({"kind": k, "color": c, "cx": cx, "cy": cy, "size": size})

        # click：直接根据 prompt 匹配颜色+形状
        for it in candidates:
            if target_kind and it.get("kind") != target_kind:
                continue
            if target_color and it.get("color") != target_color:
                continue
            tol = meta.get("tolerance") or (float(it.get("size") or 0) * 1.2 if it.get("size") else 12)
            return {"cx": it.get("cx"), "cy": it.get("cy"), "tol": tol}

        # odd：items 为 dict，目标是唯一的少数派
        if captcha.kind.lower() == "odd" and candidates:
            combo_count = {}
            for it in candidates:
                key = (it.get("kind"), it.get("color"))
                combo_count[key] = combo_count.get(key, 0) + 1
            odd_key = next((k for k, v in combo_count.items() if v == 1), None)
            if odd_key:
                for it in candidates:
                    if (it.get("kind"), it.get("color")) == odd_key:
                        tol = meta.get("tolerance") or (float(it.get("size") or 0) * 1.2 if it.get("size") else 12)
                        return {"cx": it.get("cx"), "cy": it.get("cy"), "tol": tol}
        return None

    def solve(self, captcha: CaptchaData, image_uri: Optional[str] = None) -> AgentResult:
        kind = captcha.kind.lower()
        
        # Group 1: Text/Math (Direct Answer)
        text_types = {
            "text", "math", "case_sensitive", "distorted", "reverse", 
            "num_sum", "num_count", "vowel_count", "hex2dec", "count_shape"
        }
        
        # Group 2: Coordinate (Axis + Click)
        click_types = {"click", "odd", "diff"}
        
        # Group 3: Grid (3x3 Selection)
        grid_types = {
            "grid", "gridcolor", "gridshape", "gridvowel", 
            "seq", "charseq", "arrowseq", "alphaseq", "numseq"
        }

        if kind in click_types:
            return self._solve_click(captcha, image_uri)
        elif kind in grid_types:
            return self._solve_grid(captcha, image_uri)
        else:
            # Default to text solver for known text types and any unknown types
            return self._solve_text(captcha, image_uri)

    def _solve_text(self, captcha: CaptchaData, image_uri: Optional[str] = None) -> AgentResult:
        # Use provided image_uri (PNG) if available, else fallback to captcha.data_uri (SVG)
        img_url = image_uri if image_uri else captcha.data_uri

        anagram_hint = ""
        if captcha.kind.lower() == "anagram":
            candidates = []
            try:
                words = (captcha.meta or {}).get("words") or []
                candidates = [str(w).lower() for w in words if isinstance(w, str)]
            except Exception:
                candidates = []
            anagram_hint = "\n7) This is an anagram. Use ONLY the shown letters to form a REAL, common English word. Prefer the most frequent/ordinary word; avoid rare/technical terms. Double-check the word is valid before responding."

        extra_rules = ""
        kind_lower = captcha.kind.lower()
        if kind_lower == "sumdigits":
            extra_rules = (
                "\nSPECIAL RULE for sumdigits:"
                "\n- Only sum numeric digits 0-9 that are VISIBLE in the image."
                "\n- If there are NO numeric digits, the sum is 0."
                "\n- Never convert letters to numbers or use letter positions."
                "\n- List the digits you used in reasoning before summing."
            )
        elif kind_lower == "digitcount":
            extra_rules = (
                "\nSPECIAL RULE for digitcount:"
                "\n- Count ONLY the numeric digits 0-9 that are VISIBLE in the image."
                "\n- The digit '0' counts as a digit; do NOT reinterpret it as the letter O."
                "\n- Do NOT treat letters as digits or convert letters to numbers."
                "\n- If no digits are visible, the count is 0."
                "\n- In reasoning, explicitly list the digits you counted and their total."
            )
        elif kind_lower == "vowelcount":
            extra_rules = (
                "\nSPECIAL RULE for vowelcount:"
                "\n- Count ONLY visible vowel letters A/E/I/O/U (case-insensitive)."
                "\n- NEVER reinterpret digits or other symbols as vowels (e.g., do NOT treat '1' as 'I')."
                "\n- If no vowel letters are visible, the count is 0."
                "\n- In reasoning, list the vowel letters you counted and their total."
            )
        elif kind_lower == "hex2dec":
            extra_rules = (
                "\nSPECIAL RULE for hex2dec:"
                "\n- Hex digits are 0-9 and A-F only. Ignore any prefix like 'HEX'."
                "\n- After the prefix, read the remaining characters in order and form the LONGEST hex token using only 0-9/A-F (even if there are spaces/gaps)."
                "\n- Do NOT stop early if more hex digits appear later (e.g., '9 B 7 E' should become '9B7E')."
                "\n- Do NOT mix in non-hex letters or symbols."
                "\n- NEVER reinterpret digits/letters (e.g., do not change 0->D or treat H/X as hex digits)."
                "\n- Convert that exact hex token to decimal. Show the token used in reasoning."
            )
        elif kind_lower == "count":
            extra_rules = (
                "\nSPECIAL RULE for count:"
                "\n- Count ONLY shapes that match BOTH the target color name and shape described in the prompt."
                "\n- Use the given palette names for color matching; do NOT invent alternate hues or hidden conditions."
                "\n- List each matching shape you found (color + shape) in reasoning, then give the total."
                "\n- If none match, answer 0. Do NOT add extra rules (e.g., adjacency/uniqueness) beyond the prompt."
            )

        messages = [
            {"role": "system", "content": f"""
You are a CAPTCHA solver. 
Type: {captcha.kind}
Prompt: {captcha.prompt}
Follow these rules strictly:
1) Obey the prompt literally—do NOT invent hidden tasks or alternate rules.
2) For math, use the operator shown (× ÷ + -) and compute the exact result. Do NOT switch to digit-sum or other variants unless the prompt explicitly says so.
3) For text-like tasks, output the characters exactly as required; keep order and spacing; preserve case unless the prompt allows otherwise.
4) For counting tasks, count exactly what the prompt asks in the image—no extras.
5) If unsure, choose the most direct interpretation of the visible instruction; avoid meta/trick reasoning.
6) If this is an anagram task, rearrange ONLY the shown letters into a REAL, common English word; prefer the most frequent/ordinary word and avoid rare or technical terms.{anagram_hint}
7) Consistency rule: once you have produced a valid English word using all given letters and it fits the prompt, do NOT switch to a different word unless you find a clear error in your previous answer.{extra_rules}
Output JSON only: {{"answer": "your_answer", "reasoning": "brief reasoning"}}. No extra text or markdown.
"""},
            {"role": "user", "content": [
                {"type": "text", "text": "Solve this CAPTCHA."},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]}
        ]
        
        history = []
        max_retries = 3
        for i in range(max_retries):
            # Use Pydantic model for strict schema
            resp = self._call_llm(messages, response_format=TextAnswer)
            if "error" in resp:
                return AgentResult(success=False, reasoning=resp["error"])
            
            prev_same = self._consistent_with_prev(history, resp.get("answer", ""))
            ans = str(resp.get("answer", ""))
            reasoning = resp.get("reasoning", "")
            history.append({"attempt": i+1, "answer": ans, "reasoning": reasoning})

            # anagram 自审：必须是有效英文单词；若有候选列表需命中，命中后直接通过
            if captcha.kind.lower() == "anagram" and not self._anagram_word_ok(captcha, ans):
                messages.append({"role": "assistant", "content": json.dumps(resp)})
                messages.append({"role": "user", "content": "Your answer must be a valid English word formed from the given letters. If it is not a real word, try again."})
                history.append({"attempt": i+1, "answer": ans, "self_check": "failed_word_validity"})
                continue
            if captcha.kind.lower() == "anagram" and self._anagram_word_ok(captcha, ans):
                history.append({"attempt": i+1, "answer": ans, "self_check": "passed_word_validity"})
                return AgentResult(success=True, answer=ans, reasoning=reasoning or "Anagram self-validated", history=history)

            # 自检：连续两次答案一致即视为通过
            if prev_same:
                note = reasoning or "Self-consistent across attempts"
                return AgentResult(success=True, answer=ans, reasoning=note, history=history)
            
            messages.append({"role": "assistant", "content": json.dumps(resp)})
            messages.append({"role": "user", "content": "Incorrect. Please review the image and prompt carefully and try again."})
            
        # Return the last attempt as best effort, but marked as success=False
        return AgentResult(success=False, answer=ans, reasoning="Max retries exceeded", history=history)

    def _solve_click(self, captcha: CaptchaData, image_uri: Optional[str] = None) -> AgentResult:
        target = self._resolve_click_target(captcha)
        width = int((captcha.meta or {}).get("width", 240))
        height = int((captcha.meta or {}).get("height", 120))
        # Prefer client-provided PNG (captures front-end view). Otherwise, fallback to原始 SVG。
        final_data_uri = image_uri or captcha.data_uri
        axis_instruction = f"Coordinate System: Origin (0,0) is Bottom-Left. X extends Right, Y extends Up. Image size: {width} x {height}."
        
        messages = [
            {"role": "system", "content": f"""
You are a CAPTCHA solver.
Type: {captcha.kind}
Prompt: {captcha.prompt}
{axis_instruction}
Follow these rules strictly:
1) Obey the prompt literally; pick the target shape/color/object described.
2) Use the shown coordinate system; return integer x,y for the target center within 0 ≤ x < {width}, 0 ≤ y < {height}. Do NOT switch origin or axes.
3) If multiple candidates exist, pick the one best matching the full prompt (shape + color + uniqueness).
4) No hidden tricks: do not change the task type or operator.
Output JSON only: {{"x": int, "y": int, "reasoning": "brief reasoning"}}. No extra text or markdown.
"""},
            {"role": "user", "content": [
                {"type": "text", "text": "Solve this CAPTCHA. Provide x,y coordinates."},
                {"type": "image_url", "image_url": {"url": final_data_uri}}
            ]}
        ]
        
        history = []
        max_retries = 3
        for i in range(max_retries):
            # Use Pydantic model for strict schema
            resp = self._call_llm(messages, response_format=ClickAnswer)
            if "error" in resp:
                return AgentResult(success=False, reasoning=resp["error"])
            
            prev_same = self._consistent_with_prev(history, f"{resp.get('x',0)},{resp.get('y',0)}")
            try:
                x = int(resp.get("x", 0))
                y = int(resp.get("y", 0))
                # Clamp to image bounds to avoid off-canvas clicks
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                # Model may output either bottom-left or top-left origin; try both.
                candidates = [
                    (x, y, "bl"),                          # assume bottom-left as instructed
                    (x, max(0, min(height - 1, height - 1 - y)), "tl"),  # reinterpret as top-left
                ]
                # Pick candidate that verifies or is within inferred target
                chosen = candidates[0]
                for cx, cy, origin in candidates:
                    ans_candidate = f"{cx},{cy}"
                    if target and self._point_in_target(cx, cy, target):
                        chosen = (cx, cy, origin)
                        break
                x, y, origin_used = chosen
                ans = f"{x},{y}"
            except:
                ans = "0,0"
                x, y = 0, 0

            reasoning = resp.get("reasoning", "")
            history.append({"attempt": i+1, "answer": ans, "reasoning": reasoning})

            # 自检：坐标需落在目标图形内；或连续答案一致
            if target and self._point_in_target(x, y, target):
                note = reasoning or "Point inside target region"
                return AgentResult(success=True, click_point=(x, y), reasoning=note, history=history)
            if prev_same:
                note = reasoning or "Self-consistent across attempts"
                return AgentResult(success=True, click_point=(x, y), reasoning=note, history=history)
            
            messages.append({"role": "assistant", "content": json.dumps(resp)})
            if target:
                messages.append({"role": "user", "content": "The coordinates seem outside the target shape. Re-evaluate the image and provide a point within the target figure."})
            else:
                messages.append({"role": "user", "content": "Coordinates are not within the target area. Please adjust based on the axes and target description."})

        # Return the last attempt as best effort, but marked as success=False
        return AgentResult(success=False, click_point=(x, y), reasoning="Max retries exceeded", history=history)

    def _solve_grid(self, captcha: CaptchaData, image_uri: Optional[str] = None) -> AgentResult:
        # Grid types usually require selecting indices or sequence of indices
        rows = captcha.meta.get("rows", 3)
        cols = captcha.meta.get("cols", 3)
        img_url = image_uri if image_uri else captcha.data_uri
        
        # Enhanced prompt for grid/sequence types
        prompt_detail = ""
        if captcha.kind in ["seq", "charseq", "arrowseq", "alphaseq", "numseq"]:
            # Dynamic instruction based on subtype
            seq_instr = "Follow the sequence described in the Prompt."
            if "arrow" in captcha.kind:
                seq_instr = "For the arrow sequence shown in the prompt (e.g. ↑ → ...), find the matching cells in order."
            elif "alpha" in captcha.kind or "char" in captcha.kind:
                seq_instr = "For the character sequence shown in the prompt (e.g. A → B → ...), find the matching cells in order."
            elif "num" in captcha.kind:
                seq_instr = "For the number sequence shown in the prompt (e.g. 1 → 2 → ...), find the matching cells in order."

            prompt_detail = f"""
SEQUENCE TASK - CRITICAL INSTRUCTIONS:

STEP 1 - UNDERSTAND THE GRID:
You will see a 3x3 grid with VISIBLE GRID LINES dividing the image into 9 cells.
Each cell has a SMALL BLUE NUMBER (0-8) in its TOP-LEFT corner.
This corner number IS THE CELL'S INDEX - this is what you must return!

STEP 2 - CELL INDEX REFERENCE:
   TOP ROW:    0 (left) | 1 (center) | 2 (right)
   MIDDLE ROW: 3 (left) | 4 (center) | 5 (right)
   BOTTOM ROW: 6 (left) | 7 (center) | 8 (right)

STEP 3 - SOLVE THE TASK:
{seq_instr}

STEP 4 - HOW TO READ INDICES (VERY IMPORTANT):
For EACH item in the sequence:
   a) FIRST: Locate the target content (e.g., number "1", letter "A", arrow "↑")
   b) SECOND: Look at the TOP-LEFT corner of that cell
   c) THIRD: Carefully READ the small BLUE number (0-8) - this is the INDEX
   d) FOURTH: Double-check you read the corner number correctly
   e) FIFTH: Record this index number

STEP 5 - RETURN YOUR ANSWER:
Return the corner index numbers in the order of the sequence.

⚠️ CRITICAL WARNINGS:
- DO NOT return the content shown in the cell (e.g., "1", "2", "3")
- DO return the small blue corner number (e.g., "7", "8", "2")  
- The content and the corner number are DIFFERENT things!
- Read the corner numbers VERY CAREFULLY - they are small but visible
- If unsure, describe what you see in each corner before answering

Example: If you see "1" in a cell and the corner has "7", return 7, NOT 1!
"""
        elif captcha.kind in ["grid", "gridcolor", "gridshape", "gridvowel"]:
            prompt_detail = """
Grid Selection Requirements:
- Select ALL cells that match the criteria.
- If prompt says "Blue Square", select cells that are BOTH Blue AND Square.
- Indices are 0-based, row-major. Do NOT invent alternate rules.
- STRATEGY:
  1. Scan each cell (0..8).
  2. Determine if it matches the criteria.
  3. Return list of matching indices.
"""

        messages = [
            {"role": "system", "content": f"""
You are a CAPTCHA solver that reads numbers from grid cells.
Type: {captcha.kind}
Prompt: {captcha.prompt}
Grid: {rows} rows x {cols} cols.

CRITICAL rules:
1) Each grid cell has a NUMBER in its top-left corner. That number IS the cell's index. READ it; do NOT recalc.
2) Obey the prompt literally; select only cells that meet the described criteria.
3) For sequence types, order matters exactly as stated. For multi-select types, include all matching indices.
4) No hidden tricks: do not change the task, operator, or criteria.

Example for 3x3 grid:
  Cell in top-left corner has number "0" → index is 0
  Cell in top-center has number "1" → index is 1  
  Cell in top-right has number "2" → index is 2
  Cell in middle-left has number "3" → index is 3
  ... and so on up to 8 for bottom-right

{prompt_detail}
Output JSON: {{"indices": [int, ...], "reasoning": "brief reasoning"}}
IMPORTANT: Return ONLY the JSON object. Do not include any other text.
"""},
            {"role": "user", "content": [
                {"type": "text", "text": "Solve this CAPTCHA. Provide list of indices."},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]}
        ]
        
        history = []
        max_retries = 3
        for i in range(max_retries):
            # Use Pydantic model for strict schema
            resp = self._call_llm(messages, response_format=GridAnswer)
            if "error" in resp:
                return AgentResult(success=False, reasoning=resp["error"])
            
            indices = resp.get("indices", [])
            ordered_types = {"seq", "charseq", "arrowseq", "alphaseq", "numseq"}
            seq_mode = captcha.kind.lower() in ordered_types
            k_expect = int((captcha.meta or {}).get("k") or 0)
            indices_for_ans = indices if seq_mode else sorted(indices)
            ans = ",".join(str(i) for i in indices_for_ans)
            reasoning = resp.get("reasoning", "")
            history.append({"attempt": i+1, "answer": ans, "reasoning": reasoning})

            # Validation without真值: ensure length/范围/唯一性满足题面
            n_cells = int(rows) * int(cols)
            invalid = False
            if seq_mode:
                if k_expect > 0 and len(indices) != k_expect:
                    invalid = True
                if n_cells and (not indices or any((not isinstance(v, int)) or v < 0 or v >= n_cells for v in indices)):
                    invalid = True
                if len(indices) != len(set(indices)):
                    invalid = True
                if invalid:
                    messages.append({"role": "assistant", "content": json.dumps(resp)})
                    need_len = f"exactly {k_expect} indices" if k_expect else "the correct indices in order"
                    messages.append({"role": "user", "content": f"Please return {need_len}, all within 0..{n_cells-1}, no duplicates."})
                    history.append({"attempt": i+1, "self_check": "failed_validation"})
                    continue
                # passes validation; accept
                return AgentResult(success=True, indices=indices, reasoning=reasoning or "Validated length/order", history=history)
            else:
                # grid multi-select: require in-range and dedup
                if n_cells and (not indices or any((not isinstance(v, int)) or v < 0 or v >= n_cells for v in indices)):
                    messages.append({"role": "assistant", "content": json.dumps(resp)})
                    messages.append({"role": "user", "content": f"Indices must be within 0..{n_cells-1}. Please try again."})
                    history.append({"attempt": i+1, "self_check": "failed_validation"})
                    continue
                if len(indices) != len(set(indices)):
                    messages.append({"role": "assistant", "content": json.dumps(resp)})
                    messages.append({"role": "user", "content": "Do not repeat indices; include each matching cell once."})
                    history.append({"attempt": i+1, "self_check": "failed_validation"})
                    continue
            
            # 自检：连续一致视为通过
            if self._consistent_with_prev(history[:-1], ans):
                note = reasoning or "Self-consistent across attempts"
                return AgentResult(success=True, indices=indices, reasoning=note, history=history)
            
            messages.append({"role": "assistant", "content": json.dumps(resp)})
            messages.append({"role": "user", "content": "Incorrect selection. Please review the prompt and grid carefully."})

        # Return the last attempt as best effort, but marked as success=False
        return AgentResult(success=False, indices=indices, answer=ans, reasoning="Max retries exceeded", history=history)

class ExecutionAgent:
    def execute(self, result: AgentResult, captcha_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the action based on AgentResult.
        For grid types, converts indices to coordinates if possible.
        """
        if not result.success:
            # Even if failed, try to execute the best effort result if available
            if not (result.answer or result.click_point or result.indices):
                return {"action": "none", "reason": "Identification failed and no best effort result"}
            # Proceed to execute best effort result...
        
        # 1. Click Action (Coordinate-based)
        if result.click_point:
            return {
                "action": "click", 
                "x": result.click_point[0], 
                "y": result.click_point[1]
            }
        
        # 2. Grid Action (Index-based -> Coordinate-based)
        if result.indices:
            # User requirement: "Execution Agent ... Clicks according to the coordinates provided"
            # We calculate the center coordinates for each index.
            # Assuming standard 3x3 grid if not specified in meta.
            rows = 3
            cols = 3
            width = 240 # Default width
            height = 120 # Default height
            
            if captcha_meta:
                rows = int(captcha_meta.get("rows", 3))
                cols = int(captcha_meta.get("cols", 3))
                width = int(captcha_meta.get("width", 240))
                height = int(captcha_meta.get("height", 120))
            
            coordinates = []
            for idx in result.indices:
                # 0-based index to (row, col)
                r = idx // cols
                c = idx % cols
                
                # Calculate center of the cell
                # Cell width/height
                cw = width / cols
                ch = height / rows
                
                cx = int((c + 0.5) * cw)
                cy = int((r + 0.5) * ch)
                coordinates.append({"x": cx, "y": cy})
            
            # Return action with both indices (for server verification) and coordinates (for clicking)
            return {
                "action": "grid_click",
                "indices": result.indices,
                "coordinates": coordinates,
                "value": ",".join(str(i) for i in result.indices) # Fallback value
            }
            
        # 3. Input Action (Text-based)
        return {"action": "input", "value": result.answer or ""}
