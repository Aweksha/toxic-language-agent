"""Quick accuracy + consistency check. Run: python test_cases.py"""
import time

from agent import analyze_text

# (text, expected label)
CASES = [
    ("You are so kind, thank you for helping me today!", "compliment"),
    ("Holy shit, you absolutely killed that presentation!", "spicy_compliment"),
    ("You're a fucking genius, I love you man", "spicy_compliment"),
    ("Damn girl, you look hot as hell tonight", "spicy_compliment"),
    ("You're pretty smart for a girl.", "backhanded_compliment"),
    ("Wow, you look great! Have you lost weight?", "backhanded_compliment"),
    ("Your English is really good for someone from there.", "backhanded_compliment"),
    ("I love how you don't care what people think about your clothes.", "backhanded_compliment"),
    ("The meeting is at 3pm in room B.", "neutral"),
    ("Ugh, this fucking printer is broken again.", "neutral"),
    ("You are absolutely terrible at your job. Nobody wants to work with you.", "toxic"),
    ("Shut up, you worthless idiot.", "toxic"),
    ("I know where you live and you'll regret this.", "toxic"),
]

RUNS = 2


def main():
    correct = 0
    consistent = 0
    for text, expected in CASES:
        results = []
        for _ in range(RUNS):
            results.append(analyze_text(text))
            time.sleep(4)  # stay under Groq free-tier tokens/minute
        labels = [r["label"] for r in results]
        sevs = [r["severity"] for r in results]
        ok = labels[0] == expected
        same = len(set(labels)) == 1 and len(set(sevs)) == 1
        correct += ok
        consistent += same
        mark = "✓" if ok else "✗"
        print(f"{mark} expected={expected:<22} got={labels} sev={sevs} | {text}")
        if results[0]["real_meaning"]:
            print(f"    → {results[0]['real_meaning']}")

    n = len(CASES)
    print(f"\nAccuracy: {correct}/{n}   Consistent across {RUNS} runs: {consistent}/{n}")


if __name__ == "__main__":
    main()
