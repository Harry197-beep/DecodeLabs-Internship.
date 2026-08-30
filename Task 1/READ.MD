# Decode-labs-chatbot
project submission for my internship in decode labs of a limited chatbot.

# Project 1: Rule-Based AI Chatbot
**DecodeLabs Internship — Batch 2026**

## Overview

A simple rule-based chatbot that responds to predefined user inputs using
dictionary lookups instead of a long if-elif chain. This is the "foundation"
project before moving into probabilistic/generative AI — the goal is to
demonstrate control flow, decision-making logic, and clean handling of
unexpected input.

## How It Works

The bot runs in a continuous loop, reading user input, cleaning it up, and
matching it against a knowledge base of known phrases. If nothing matches,
it falls back to a default response instead of crashing or going silent.

### 1. Input Loop
```python
while True:
    raw_input_text = input("You: ")
    ...
```
Keeps the conversation running indefinitely. The only way out is the exit
command, which calls `break`.

### 2. Sanitization
```python
def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()
```
Normalizes input so `"Hello"`, `"HELLO"`, and `"  hello  "` all match the
same rule. Without this, the bot would treat every casing/whitespace
variation as a completely different input.

### 3. Knowledge Base (dictionary, not if-elif)
```python
responses = {
    "hello": "Hi there! I'm Ledger, your friendly rule-based bot. 🤖",
    "hi": "Hey! What can I help you with?",
    "how are you": "I'm just a bunch of if-else logic, but I'm doing great!",
    "what is your name": "I'm Ledger, nice to meet you.",
    "help": "Try saying: hello, how are you, what is your name, or bye.",
    "bye": "Goodbye! Thanks for chatting.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}
```
A dictionary gives O(1) lookup time no matter how many rules you add. An
if-elif chain gets slower and harder to maintain as it grows (linear
complexity, cascading failures on typos in the chain).

### 4. Matching Logic
```python
def get_response(clean_input: str) -> str:
    if clean_input in responses:
        return responses[clean_input]

    for keyword, reply in responses.items():
        if keyword in clean_input:
            return reply

    return FALLBACK_RESPONSE
```
Two passes:
1. **Exact match** — fast path, checks if the cleaned input is a key.
2. **Loose match** — checks if any known keyword appears *inside* a longer
   sentence (e.g. `"hello there, how's it going"` still matches `"hello"`).

If neither pass finds anything, it returns the fallback response instead of
throwing an error.

### 5. Exit Strategy
```python
EXIT_COMMANDS = {"exit", "quit", "bye"}

if clean_input in EXIT_COMMANDS:
    print(f"{BOT_NAME}: Goodbye! 👋")
    break
```
A clean, explicit way to end the loop — no forced kills, no crashes.

## How to Run

```bash
python chatbot.py
```

Type messages at the `You:` prompt. Type `exit`, `quit`, or `bye` to end
the chat.

## Example Inputs to Test

| Category | Example |
|---|---|
| Exact match | `hello`, `how are you`, `help` |
| Loose match | `hello there, how's it going` |
| Case/whitespace | `HELLO`, `  Hi  ` |
| Fallback | `what's the weather today` |
| Exit | `bye`, `exit`, `quit` |

## Key Concepts Demonstrated

- Control flow (`while`, `if`/`else`)
- Dictionary-based lookup vs. if-elif ladders
- Input normalization
- Fallback/default handling
- Clean program termination

## Possible Extensions

- Multiple varied replies per intent (list instead of single string)
- Basic memory (e.g. remembering the user's name across turns)
- Splitting the knowledge base into its own file (`knowledge_base.py`)
- Passing unmatched input to an LLM as a hybrid rule + generative system
