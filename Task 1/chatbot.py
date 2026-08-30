"""
DecodeLabs - Project 1: Rule-Based AI Chatbot
Batch 2026

Design notes (per the brief):
- Uses a dictionary ("knowledge base") instead of an if-elif ladder.
  Dict lookups are O(1) regardless of how many rules you add; an
  if-elif chain is O(n) and gets messy fast (the "Anti-Pattern" slide).
- Sanitization: input is lowercased + stripped before matching, so
  "Hello", "HELLO", " hello " all hit the same rule.
- Fallback: unmatched input never crashes the bot - it gets a
  default response via .get(key, fallback).
- Exit: typing "exit" (or bye/quit) breaks the while loop cleanly.
"""

BOT_NAME = "Ledger"  # give it a bit of personality

# --- Knowledge base: keyword -> response ---
# Add more entries any time - the lookup logic never changes.
responses = {
    "hello": "Hi there! I'm Ledger, your friendly rule-based bot. 🤖",
    "hi": "Hey! What can I help you with?",
    "how are you": "I'm just a bunch of if-else logic, but I'm doing great!",
    "what is your name": f"I'm {BOT_NAME}, nice to meet you.",
    "help": "Try saying: hello, how are you, what is your name, or bye.",
    "bye": "Goodbye! Thanks for chatting.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

EXIT_COMMANDS = {"exit", "quit", "bye"}
FALLBACK_RESPONSE = "I do not understand that yet. Try 'help' to see what I know."


def sanitize(raw_input: str) -> str:
    """Normalize user input: lowercase + strip whitespace."""
    return raw_input.lower().strip()


def get_response(clean_input: str) -> str:
    """
    Look up the cleaned input in the knowledge base.
    Falls back to a default response if there's no exact match,
    then tries a loose keyword match as a second pass.
    """
    # 1. Exact match (fast path)
    if clean_input in responses:
        return responses[clean_input]

    # 2. Loose match: does any known keyword appear in the sentence?
    for keyword, reply in responses.items():
        if keyword in clean_input:
            return reply

    # 3. Nothing matched
    return FALLBACK_RESPONSE


def main():
    print(f"{BOT_NAME}: Hello! Type 'exit' anytime to end our chat.\n")

    while True:  # The Heartbeat - keeps running until the kill command
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! 👋")
            break

        reply = get_response(clean_input)
        print(f"{BOT_NAME}: {reply}")


if __name__ == "__main__":
    main()
