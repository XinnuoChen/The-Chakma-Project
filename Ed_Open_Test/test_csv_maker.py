import csv
import random
from datetime import date, timedelta

random.seed(7)

NAMES = ["Amina", "Jon", "Sara", "Khalid", "Mina", "Ed", "Noah", "Lina"]
PLACES = ["London", "Edinburgh", "the office", "the station", "home", "the bank"]
PRODUCTS = ["NatWest app", "payment", "invoice", "account", "card", "transfer"]
ACTIONS = ["sent", "received", "cancelled", "confirmed", "scheduled", "delayed"]
TEMPLATES = [
    "Please translate this message exactly as written.",
    "{name} {action} the {product} at {time}.",
    "The total is £{amt} and it’s due on {d}.",
    "Meet me at {place} at {time}, please.",
    "I can’t log in — the code {code} isn’t working.",
    "Reminder: your appointment is on {d} at {time}.",
    "If this fails, try again after {mins} minutes.",
    "Do not translate `CODE_{code}` or URLs like https://example.com/{code}.",
    "She said: “{quote}”",
    "Send {qty} items to {place}.",
]

QUOTES = [
    "I’ll be there in 5 minutes.",
    "Can you call me back?",
    "That price is too high.",
    "Let’s do it tomorrow.",
]

def rand_date():
    start = date(2026, 2, 1)
    d = start + timedelta(days=random.randint(0, 60))
    return d.isoformat()

def rand_time():
    h = random.randint(7, 22)
    m = random.choice([0, 15, 30, 45])
    return f"{h:02d}:{m:02d}"

def make_row(i):
    t = random.choice(TEMPLATES)
    text = t.format(
        name=random.choice(NAMES),
        action=random.choice(ACTIONS),
        product=random.choice(PRODUCTS),
        place=random.choice(PLACES),
        time=rand_time(),
        d=rand_date(),
        amt=random.choice([9.99, 12.50, 48.00, 105.75, 420.00]),
        code=random.randint(1000, 9999),
        mins=random.choice([5, 10, 15, 30, 60]),
        quote=random.choice(QUOTES),
        qty=random.randint(1, 12),
    )
    return {
        "id": f"seed_{i:04d}",
        "domain": "general",
        "en": text,
        "tgt": "",
        "source": "seed",
        "notes": ""
    }

def build_seed_csv(path="parallel_seed.csv", n=200):
    rows = [make_row(i) for i in range(1, n+1)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {n} rows to {path}")

if __name__ == "__main__":
    build_seed_csv(n=200)
