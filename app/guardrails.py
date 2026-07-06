"""
Guardrails: input and output safety checks that wrap every request through
the agent gateway.

Why guardrails matters, explanation:
- Input guardrails prevent the agent from being weaponised against its users
  (prompt injection) or from leaking sensitive user data to the LLM
  unnecessarily (PII).
- Output guardrails catch the cases where the model ignores the input
  guardrails, hallucinates harmful content, or leaks data it was given as
  context.
- These are intentionally layered and independent so each one can be
  toggled via config without touching the others.

The PII detection uses Presidio with pattern-only recognizers (regex + rule
engines) -- this runs fully in-process with no extra API call, no network
dependency, and no model download. For production you would layer a proper
NLP-based recognizer on top (Presidio supports spaCy, Hugging Face, etc.),
but regex gets the high-value cases (SSN, credit card, email, phone) reliably
and cheaply, which is usually the right trade-off to start with.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer

from app.config import settings

logger = logging.getLogger(settings.app_name)

# INPUT PATTERNS

_INJECTION_PATTERNS = [
    r"ignore\s+(\w+\s+)?(previous|above|all|prior)\s+(\w+\s+)?instructions?",
    r"disregard\s+(your|all|previous|prior|any)\s+",
    r"you\s+are\s+now\s+(a|an|the|\w+)\s*\w*",        # matches "you are now DAN" (no article)
    r"new\s+system\s+prompt",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"forget\s+(\w+\s+)*(training|instructions|rules|guidelines)",  # matches "forget all your training"
    r"act\s+as\s+if\s+you\s+(have\s+no|don.t\s+have)\s+",
    r"override\s+(your\s+)?(safety|ethical|content)\s+(filter|guideline|restriction)",
]

_BLOCKED_PATTERNS = [
    r"\b(make|build|create|synthesize|produce|detonate)\s+\w{0,10}\s*(bomb|explosive\s+device|bioweapon|nerve\s+agent)\b",
    r"\b(ransomware|malware|rootkit|keylogger|spyware)\b",
    r"\bhack\s+(into|their|the|a)\s+\w+\s*(server|system|database|network)\b",
    r"\bchild\s+(sexual\s+abuse|pornography|exploitation)\b",
    r"\bsuicide\s+(bomb|vest|attack)\b",
]

_PII_RECOGNIZERS = [
    PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[Pattern("ssn", r"\b\d{3}-\d{2}-\d{4}\b", 0.85)],
    ),
    PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.85)],
    ),
    PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[Pattern("phone", r"\b(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b", 0.75)],
    ),
    PatternRecognizer(
        supported_entity="CREDIT_CARD",
        patterns=[Pattern("cc", r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", 0.85)],
    ),
]

# OUTPUT PATTERNS

_OUTPUT_UNSAFE_PATTERNS = [
    r"(step.by.step|detailed)\s+(instructions?|guide|tutorial)\s+(to|for|on\s+how\s+to)\s+"
    r"(make|build|create|synthesize)\s+\w{0,10}\s*(bomb|weapon|malware|exploit)",
    r"(detailed|step.by.step)\s+instructions?\s+(for|on|to)\s+(how\s+to\s+)?"
    r"(make|create|build|write|deploy)\s+\w*\s*(malware|bomb|exploit|ransomware|weapon)",
    r"\b(here\s+is|here.s)\s+(how\s+to|the\s+(code|script|payload)\s+for)\s+(hack|exploit|attack)\b",
]

# RESULT DATACLASS

@dataclass
class GuardrailResult:
    blocked: bool
    reason: Optional[str] = None
    pii_entities: List[str] = field(default_factory=list)

def check_input(input_text: str) -> GuardrailResult:
    """
    Run all input guardrails on the user message.
    
    Return a GuardrailResult where `blocked=True` means the request
    should not be forwarded to the agent. Check run in cheapest first
    order, (Regex before pattern recognizers) so that expensive work is
    skipped as soon as a block is found.
    """

    if not settings.guardrails_enabled:
        return GuardrailResult(blocked=False)
    
    if settings.injection_detection_enabled:
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                logger.warning(f"Input guardrail blocked due to injection pattern: {pattern}")
                return GuardrailResult(
                    blocked=True,
                    reason="Your message appears to contain a prompt injection attempt and cannot be processed."
                )
    
    if settings.blocked_topics_enabled:
        for pattern in _BLOCKED_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                logger.warning(f"Input guardrail blocked due to blocked topic pattern: {pattern}")
                return GuardrailResult(
                    blocked=True,
                    reason="Your message appears to contain a blocked topic and cannot be processed."
                )
            
    if settings.pii_detection_enabled:
        entities_found = []
        for recognizer in _PII_RECOGNIZERS:
            results = recognizer.analyze(text=input_text, entities=[recognizer.supported_entities[0]])
            if results:
                entities_found.append(recognizer.supported_entities[0])

        if entities_found:
            logger.warning("PII detected in input: %s", entities_found)
            return GuardrailResult(
                blocked=True,
                reason=(
                    f"Your message appears to contain sensitive personal information "
                    f"({', '.join(entities_found)}). Please remove it before sending."
                ),
                pii_entities=entities_found,
            )
            
    return GuardrailResult(blocked=False)

def check_output(text: str) -> GuardrailResult:
    """Run output safety checks on the agent's response.

    If this returns blocked=True the gateway will replace the response
    with a safe fallback message rather than returning harmful content.
    """
    if not settings.guardrails_enabled or not settings.output_safety_enabled:
        return GuardrailResult(blocked=False)

    for pattern in _OUTPUT_UNSAFE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("Unsafe content detected in output, suppressing response.")
            return GuardrailResult(
                blocked=True,
                reason="The response was flagged by output safety filters and has been suppressed.",
            )

    return GuardrailResult(blocked=False)
    
