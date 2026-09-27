"""
magicpin AI Challenge — Vera bot
=================================
Implements the 5 required endpoints (challenge-testing-brief.md §2):
  POST /v1/context   - receive category/merchant/customer/trigger context
  POST /v1/tick       - periodic wake-up; bot may initiate messages
  POST /v1/reply      - respond to a merchant/customer reply
  GET  /v1/healthz    - liveness probe
  GET  /v1/metadata   - bot identity

Run locally:
    export GROQ_API_KEY=gsk_...
    uvicorn bot:app --host 0.0.0.0 --port 8080

Test locally:
    export BOT_URL=http://localhost:8080
    python judge_simulator.py

Deploy: push this + requirements.txt to a GitHub repo, connect it to
Render.com or Railway.app as a web service. Set GROQ_API_KEY as
an environment variable in the platform's dashboard (never commit it).

Uses Groq (https://console.groq.com) instead of a paid provider — Groq
issues free API keys with no card required, and it's one of the
providers judge_simulator.py itself already supports (LLM_PROVIDER =
"groq"). Groq's API is OpenAI-compatible, so this uses the `openai`
python package pointed at Groq's base_url.
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
START = time.time()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"  # free-tier Groq model; fast + capable enough for this task

# ---------------------------------------------------------------------------
# In-memory state (fine per the brief — no restarts during the test window)
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}      # (scope, context_id) -> {"version": int, "payload": dict}
conversations: dict[str, list[dict]] = {}       # conversation_id -> [{"from": "...", "msg": "..."}]
sent_bodies: dict[str, set[str]] = {}           # conversation_id -> set of bodies already sent (anti-repetition)


def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# ---------------------------------------------------------------------------
# The composer — this is the part worth spending your time on.
# Everything else in this file is plumbing you shouldn't need to touch much.
# ---------------------------------------------------------------------------

COMPOSER_SYSTEM_PROMPT = """You are composing ONE WhatsApp message on behalf of "Vera", \
magicpin's merchant-marketing assistant. Follow these rules exactly:

1. Anchor on a concrete, verifiable fact from the provided contexts (a number, date, \
headline, or peer stat). Never say generic things like "increase your sales" or "10% off" \
if a specific offer/number is available instead.
2. Match the category's voice profile exactly — vocabulary allowed, taboos forbidden. \
Clinical/peer categories (dentists, doctors) must NOT sound promotional.
3. Personalize to the specific merchant's numbers, offers, and conversation history given.
4. State clearly, in the message, why you're reaching out now (the trigger).
5. Use exactly ONE compulsion lever from: specificity, loss aversion, social proof, \
effort externalization, curiosity, reciprocity, asking the merchant a question, or a \
single binary CTA. Do not stack multiple CTAs.
6. Match the merchant's language preference. Hindi-English code-mix is fine and often \
preferred — do not force pure English if the merchant's language field says otherwise.
7. Never invent data not present in the given contexts. No fake citations, no fake \
competitor names, no fake numbers.
8. Keep it concise — no long preambles, no re-introducing yourself after the first message.
9. Never repeat a message verbatim that you can see was already sent in this conversation.

Return ONLY a JSON object with these exact keys, nothing else, no markdown fences:
{"body": "...", "cta": "open_ended" | "binary" | "none", "send_as": "vera" | "merchant_on_behalf", "suppression_key": "...", "rationale": "..."}
"""


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None,
            prior_messages: list[str] | None = None) -> dict:
    """Core composer. Called both from /v1/tick (new message) and indirectly
    to build submission.jsonl. Deterministic: temperature=0."""

    user_payload = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
        "already_sent_in_this_conversation": prior_messages or [],
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {
            "body": "",
            "cta": "none",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": f"LLM call failed ({e}); returning empty action.",
        }

    # strip accidental markdown fences just in case
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # fail safe rather than returning malformed JSON to the judge
        result = {
            "body": "",
            "cta": "none",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "LLM output failed to parse as JSON; returning empty action.",
        }
    return result


def compose_reply(conversation_id: str, merchant_message: str,
                   prior_turns: list[dict]) -> dict:
    """Used by /v1/reply. Decides send / wait / end."""
    system = """You are Vera, continuing a WhatsApp conversation with a merchant (or \
their customer). Given the conversation so far and their latest message, decide the \
next move.

Detect auto-replies: if the merchant's message looks like a canned WhatsApp Business \
auto-reply (generic "thank you for contacting us / will forward to team" phrasing), try \
ONE more direct nudge, then gracefully end if it recurs.

Detect explicit intent ("yes", "let's do it", "I want to join") and route straight to \
action — never go back to qualifying questions after explicit agreement.

Detect a clear "not interested" / "stop" and end gracefully and politely.

Return ONLY JSON, no markdown fences, with exactly one of these shapes:
{"action": "send", "body": "...", "cta": "...", "rationale": "..."}
{"action": "wait", "wait_seconds": 1800, "rationale": "..."}
{"action": "end", "rationale": "..."}
"""
    payload = {"conversation_so_far": prior_turns, "latest_message": merchant_message}
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=400, temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"action": "end", "rationale": f"LLM call failed ({e}); ending safely."}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"action": "end", "rationale": "LLM output failed to parse; ending safely."}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "CHANGE_ME",
        "team_members": ["Ravi Vijay Thakur"],
        "model": MODEL,
        "approach": "Single-prompt composer over the 4-context framework; rule-based reply router with auto-reply and intent detection.",
        "contact_email": "CHANGE_ME@example.com",
        "version": "0.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        trg = get_ctx("trigger", trg_id)
        if not trg:
            continue

        merchant_id = trg.get("payload", {}).get("merchant_id") or trg.get("merchant_id")
        if not merchant_id:
            continue

        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug")
        category = get_ctx("category", category_slug) if category_slug else None
        if not category:
            continue

        customer_id = trg.get("customer_id")
        customer = get_ctx("customer", customer_id) if customer_id else None

        conversation_id = f"conv_{merchant_id}_{trg_id}"
        prior = [t["msg"] for t in conversations.get(conversation_id, []) if t["from"] in ("vera", "merchant_on_behalf")]

        result = compose(category, merchant, trg, customer, prior_messages=prior)

        if not result.get("body"):
            continue  # restraint is fine — skip if the composer had nothing worth sending

        already_sent = sent_bodies.setdefault(conversation_id, set())
        if result["body"] in already_sent:
            continue  # anti-repetition guard
        already_sent.add(result["body"])

        conversations.setdefault(conversation_id, []).append(
            {"from": result.get("send_as", "vera"), "msg": result["body"]}
        )

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": result["body"],
            "cta": result.get("cta", "none"),
            "suppression_key": result.get("suppression_key", trg.get("suppression_key", "")),
            "rationale": result.get("rationale", ""),
        })

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    prior_turns = conversations.setdefault(body.conversation_id, [])
    prior_turns.append({"from": body.from_role, "msg": body.message})

    result = compose_reply(body.conversation_id, body.message, prior_turns)

    if result.get("action") == "send" and result.get("body"):
        already_sent = sent_bodies.setdefault(body.conversation_id, set())
        if result["body"] in already_sent:
            # force a graceful end rather than repeat verbatim
            result = {"action": "end", "rationale": "Avoiding verbatim repeat; ending conversation."}
        else:
            already_sent.add(result["body"])
            prior_turns.append({"from": "vera", "msg": result["body"]})

    return result
