# Vera Challenge Submission — [Your Team Name]

## Approach
A single-prompt composer (Claude, temperature=0) takes the four context layers
(category, merchant, trigger, optional customer) as structured JSON and returns
a strict-JSON message object. The system prompt encodes the category voice
rules, the compulsion-lever list, and the anti-fabrication constraint directly
from the challenge brief, rather than relying on the model's own judgment of
what a "good WhatsApp message" looks like.

For multi-turn replies, a second prompt classifies the merchant's reply into
send / wait / end, with explicit instructions to detect auto-replies (repeated
canned phrasing) and explicit intent ("yes, let's do it") so the bot doesn't
lose momentum by re-qualifying after an agreement.

## Tradeoffs
- [Fill in: e.g. "Used a single model call rather than a retrieval step over
  digest items, given the time constraint — for categories with a large
  digest list this may under-use older-but-relevant items."]
- [Fill in: e.g. "Anti-repetition is a simple exact-string check per
  conversation; a paraphrase-repeat wouldn't be caught."]
- [Fill in anything else you cut for time.]

## What additional context would have helped most
- [Fill in: e.g. "A canonical list of the 30 test pairs in the dataset zip
  itself, rather than needing to infer them — cost us time during setup."]
- [Fill in anything else.]
