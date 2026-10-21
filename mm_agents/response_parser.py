import json
import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger("desktopenv.agent")


def _evidence_body(response):
    match = re.search(
        r"<evidence>\s*(.*?)\s*</?evidence>",
        response or "", re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None


def extract_visual_anchor(response):
    """Return an (x, y, width, height) pixel region, or None if invalid."""
    body = _evidence_body(response)
    anchors = re.findall(r"<visual_anchor\b[^>]*>", body or "", re.IGNORECASE)
    if not anchors:
        return None
    if len(anchors) > 1:
        logger.warning("Multiple visual_anchor tags; using the first one")
    try:
        element = ET.fromstring(anchors[0])
        region = tuple(int(element.attrib[name]) for name in ("x", "y", "width", "height"))
        x, y, width, height = region
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("origin must be non-negative and size must be positive")
        return region
    except (ET.ParseError, KeyError, ValueError) as error:
        logger.warning("Ignoring invalid visual_anchor: %s", error)
        return None


def _extract_json_objects(text):
    """Extract adjacent or fenced JSON objects without requiring one JSON document."""
    decoder = json.JSONDecoder()
    objects = []
    cursor = 0
    text = text or ""
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict) and "evidence_type" in value:
            objects.append((value, start, start + consumed))
        cursor = start + consumed
    return objects


def extract_evidence(response):
    """Return evidence text without visual anchors or legacy screenshot JSON."""
    text = _evidence_body(response)
    if text is None:
        return None
    text = re.sub(r"</?visual_anchor\b[^>]*>", "", text, flags=re.IGNORECASE)
    for _, start, end in reversed(_extract_json_objects(text)):
        text = text[:start] + text[end:]
    text = re.sub(r"```(?:json)?\s*```", "", text, flags=re.IGNORECASE)
    return text.strip() or None


def parse_pyautogui_response(response):
    """Parse optional evidence SCREENSHOT plus Python actions.

    SCREENSHOT is deliberately represented as a structured host action rather
    than Python code, so the runner can capture and persist it safely.
    """
    response = response or ""
    evidence_body = _evidence_body(response)
    action_match = re.search(
        r"<action>\s*(.*?)\s*</action>", response, re.IGNORECASE | re.DOTALL
    )
    evidence_actions = []
    if evidence_body is not None:
        evidence_actions = [
            value for value, _, _ in _extract_json_objects(evidence_body)
            if value.get("evidence_type") == "SCREENSHOT"
        ]

    structured = re.search(r"<(?:think|evidence|analysis|action)\b", response, re.IGNORECASE)
    action_text = action_match.group(1) if action_match else ("" if structured else response)
    action_text = (action_text or "").strip()

    fenced = re.fullmatch(
        r"```(?:python|py)?\s*(.*?)```", action_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    python_code = fenced.group(1).strip() if fenced else action_text
    python_actions = [python_code] if python_code else []

    return evidence_actions + python_actions