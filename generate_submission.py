"""
Generates submission.jsonl (challenge-brief.md §7.2) by calling bot.py's
compose() function directly for each of the 30 canonical test pairs.

BEFORE RUNNING:
1. Download the dataset zip from the challenge portal and unzip it next
   to this script, so you have:
       dataset/categories/*.json
       dataset/merchants/*.json
       dataset/customers/*.json
       dataset/triggers/*.json
2. Find the file that lists the 30 canonical (merchant, trigger) test pairs
   — it should be in the dataset zip (commonly named test_set.json,
   test_pairs.json, or similar) or on the challenge portal page.
   Point TEST_PAIRS_FILE at it below. If you can't find one, the fallback
   below builds pairs by matching triggers to their merchant_id — check
   with the organizers that this matches their canonical 30.
3. export ANTHROPIC_API_KEY=sk-ant-...
4. python generate_submission.py
"""

import json
import os
from pathlib import Path

from bot import compose  # reuses the exact same composer bot.py uses at runtime

DATASET_DIR = Path(__file__).parent / "dataset"
TEST_PAIRS_FILE = DATASET_DIR / "test_set.json"  # <-- CHANGE if the real file has a different name
OUTPUT_FILE = Path(__file__).parent / "submission.jsonl"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all(subdir: str) -> dict[str, dict]:
    out = {}
    folder = DATASET_DIR / subdir
    for f in folder.glob("*.json"):
        data = load_json(f)
        key = data.get("slug") or data.get("merchant_id") or data.get("customer_id") or data.get("id") or f.stem
        out[key] = data
    return out


def main():
    categories = load_all("categories")
    merchants = load_all("merchants")
    customers = load_all("customers")
    triggers = load_all("triggers")

    if TEST_PAIRS_FILE.exists():
        test_pairs = load_json(TEST_PAIRS_FILE)
        # expected shape: [{"test_id": "T01", "merchant_id": "...", "trigger_id": "...", "customer_id": null}, ...]
    else:
        print(f"WARNING: {TEST_PAIRS_FILE} not found. Falling back to auto-pairing "
              f"the first 30 triggers with their referenced merchant. "
              f"Confirm with organizers this matches the real canonical set.")
        test_pairs = []
        for i, (trg_id, trg) in enumerate(list(triggers.items())[:30]):
            merchant_id = trg.get("payload", {}).get("merchant_id") or trg.get("merchant_id")
            test_pairs.append({
                "test_id": f"T{i+1:02d}",
                "merchant_id": merchant_id,
                "trigger_id": trg_id,
                "customer_id": trg.get("customer_id"),
            })

    results = []
    for pair in test_pairs:
        merchant = merchants.get(pair["merchant_id"])
        trigger = triggers.get(pair["trigger_id"])
        if not merchant or not trigger:
            print(f"SKIP {pair['test_id']}: missing merchant or trigger data")
            continue

        category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug")
        category = categories.get(category_slug)
        if not category:
            print(f"SKIP {pair['test_id']}: no category '{category_slug}' found")
            continue

        customer = customers.get(pair.get("customer_id")) if pair.get("customer_id") else None

        result = compose(category, merchant, trigger, customer)

        results.append({
            "test_id": pair["test_id"],
            "body": result.get("body", ""),
            "cta": result.get("cta", "none"),
            "send_as": result.get("send_as", "vera"),
            "suppression_key": result.get("suppression_key", trigger.get("suppression_key", "")),
            "rationale": result.get("rationale", ""),
        })
        print(f"OK {pair['test_id']}: {result.get('body', '')[:80]}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
