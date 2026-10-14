"""Deterministic guardrails (fast, cheap, auditable) run before and after the LLM agents.

Input stage
  * size limits
  * sensitive data (card numbers validated with Luhn, CVV, CPF, passwords) -> REDACTED before
    reaching any LLM or log, and the user is warned
  * prompt-injection / jailbreak attempts -> blocked
  * clearly harmful or fraud-enabling requests -> blocked
  * attempts to reach another merchant's data -> flagged (tools are server-scoped anyway)
Output stage
  * redact any sensitive data that might appear in a generated answer
  * enforce non-empty, bounded answers
Relevance (off-topic) is handled semantically by the Router's ``out_of_scope`` route.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000
MAX_OUTPUT_CHARS = 4000

_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_CPF = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CVV = re.compile(r"\b(cvv|cvc|c[oó]digo de seguran[cç]a|security code)\b\D{0,15}\d{3,4}\b", re.IGNORECASE)
# A secret value must follow an explicit assignment and look like a secret (has a digit/symbol),
# so sentences like "the PIN is incorrect" are not redacted.
_PASSWORD = re.compile(
    r"\b(senha|password|passwd|pin)\b\s*(?:[:=]|é|eh|is)\s*([^\s]*[\d@#$%!&*][^\s]*)", re.IGNORECASE)

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"ignor[ae] (todas )?as instru[cç][oõ]es",
    r"disregard (your|the) (instructions|system prompt|rules)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system|hidden) (prompt|instructions)",
    r"mostre (o|seu) prompt",
    r"you are now (?!talking)",
    r"\bact as (an? )?(unrestricted|jailbroken|dan)\b",
    r"\bdeveloper mode\b|\bjailbreak\b|\bDAN mode\b",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_HARMFUL = [
    r"\b(clon\w+|skimm\w+|chupa[- ]?cabra)\b.*\b(card|cart[aã]o|cartoes|cards)\b",
    r"\b(card|cart[aã]o)\b.*\b(clon\w+|skimm\w+)\b",
    r"\b(bypass|burlar|contornar|driblar)\b.*\b(antifraud\w*|antifraude|fraud|fraude|3ds|security)\b",
    r"\b(lavar|launder\w*)\b.*\b(dinheiro|money)\b",
    r"\b(stolen|roubad\w+) (card|cart[aã]o) (numbers|n[uú]meros)\b",
    r"\bfake (chargeback|charge ?back)\b|\bchargeback fraud\b",
]
_OTHER_MERCHANT = re.compile(r"\bcliente\d{3,}\b", re.IGNORECASE)


@dataclass
class GuardrailResult:
    text: str
    blocked: bool = False
    block_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    user_notice: str | None = None


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _cpf_ok(cpf: str) -> bool:
    nums = [int(c) for c in re.sub(r"\D", "", cpf)]
    if len(nums) != 11 or len(set(nums)) == 1:
        return False
    for i in (9, 10):
        s = sum(nums[j] * ((i + 1) - j) for j in range(i))
        if (s * 10 % 11) % 10 != nums[i]:
            return False
    return True


def redact_sensitive(text: str, *, include_secrets: bool = True) -> tuple[str, list[str]]:
    flags: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            flags.append("pii_card_number")
            return f"[CARD ****{digits[-4:]}]"
        return m.group(0)

    def cpf(m: re.Match) -> str:
        if _cpf_ok(m.group(0)):
            flags.append("pii_cpf")
            return "[CPF REDACTED]"
        return m.group(0)

    text = _CARD_CANDIDATE.sub(card, text)
    text = _CPF.sub(cpf, text)
    if not include_secrets:
        return text, sorted(set(flags))
    if _CVV.search(text):
        flags.append("pii_cvv")
        text = _CVV.sub(lambda m: f"{m.group(1)} [REDACTED]", text)
    if _PASSWORD.search(text):
        flags.append("pii_password")
        text = _PASSWORD.sub(lambda m: f"{m.group(1)} [REDACTED]", text)
    return text, sorted(set(flags))


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def check_input(message: str, user_id: str) -> GuardrailResult:
    stripped = (message or "").strip()
    if not stripped:
        return GuardrailResult(text="", blocked=True, block_reason="empty_message", flags=["empty"])
    if len(stripped) > MAX_INPUT_CHARS:
        return GuardrailResult(text=stripped[:MAX_INPUT_CHARS], blocked=True, block_reason="message_too_long",
                               flags=["too_long"])

    text, flags = redact_sensitive(stripped)
    norm = _norm(text)

    if any(re.search(p, norm, re.IGNORECASE) for p in _INJECTION):
        return GuardrailResult(text=text, blocked=True, block_reason="prompt_injection", flags=[*flags, "prompt_injection"])
    if any(re.search(p, norm, re.IGNORECASE) for p in _HARMFUL):
        return GuardrailResult(text=text, blocked=True, block_reason="harmful_request", flags=[*flags, "harmful"])

    others = {m.lower() for m in _OTHER_MERCHANT.findall(norm)} - {user_id.lower()}
    if others:
        flags.append("other_merchant_reference")

    notice = None
    if any(f.startswith("pii_") for f in flags):
        notice = "pii"
    return GuardrailResult(text=text, flags=flags, user_notice=notice)


def check_output(answer: str) -> GuardrailResult:
    # Answers legitimately talk about "PIN"/"senha" in instructions; only redact hard identifiers.
    text, flags = redact_sensitive(answer or "", include_secrets=False)
    flags = [f"output_{f}" for f in flags]
    if not text.strip():
        flags.append("empty_output")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + " …"
        flags.append("output_truncated")
    return GuardrailResult(text=text, flags=flags)


BLOCK_MESSAGES = {
    "empty_message": {"en": "Please type your question so I can help.", "pt": "Por favor, digite sua pergunta para que eu possa ajudar."},
    "message_too_long": {"en": f"Your message is too long (max {MAX_INPUT_CHARS} characters). Could you summarise it?",
                         "pt": f"Sua mensagem é muito longa (máx. {MAX_INPUT_CHARS} caracteres). Pode resumir?"},
    "prompt_injection": {"en": "I can't follow instructions that try to change how I work. I'm happy to help with Getnet products or your account.",
                         "pt": "Não posso seguir instruções que tentam alterar meu funcionamento. Posso ajudar com produtos Getnet ou com a sua conta."},
    "harmful_request": {"en": "I can't help with that request. If you suspect fraud on your account, I can connect you with Getnet's team.",
                        "pt": "Não posso ajudar com esse pedido. Se você suspeita de fraude na sua conta, posso te conectar com a equipe da Getnet."},
}

PII_NOTICE = {
    "en": "For your security, never share full card numbers, CVV, passwords or documents in chat - I've hidden that information.",
    "pt": "Para sua segurança, nunca compartilhe números completos de cartão, CVV, senhas ou documentos no chat - ocultei essas informações.",
}


def guess_language(text: str) -> str:
    t = _norm(text)
    pt_hits = len(re.findall(r"\b(o|a|um|uma|de|do|da|com|que|não|nao|minha|meu|como|quando|você|voce|para|"
                             r"quero|preciso|falar|vendas|maquininha|oi|olá|ola|é|está|qual|quais|posso)\b", t))
    en_hits = len(re.findall(r"\b(the|my|what|how|when|is|are|do|does|can|you|your|i)\b", t))
    return "pt" if pt_hits > en_hits else "en"
