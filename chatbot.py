import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Config ---
PERSONA = "pirate who loves Python" # "strict interviewer preparing me for a FAANG interview" # "chef who explains everything using cooking metaphors" # "senior Python backend engineer"  # change this to experiment

SYSTEM_PROMPT = f"""You are a {PERSONA}.
You give clear, concise answers.
You remember everything said earlier in the conversation.
When you don't know something, you say so honestly."""

MAX_HISTORY = 10  # keep last 10 exchanges to control token usage

def chat(messages: list, user_input: str) -> str:
    """Send user input, get reply, return reply text."""

    messages.append({"role": "user", "content": user_input})

    # Trim history if too long — keep system prompt + last MAX_HISTORY messages
    history = [messages[0]] + messages[-(MAX_HISTORY):]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        messages=history
    )

    reply = response.choices[0].message.content

    # Save assistant reply to history
    messages.append({"role": "assistant", "content": reply})

    # Show token usage
    usage = response.usage
    print(f"\n[tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out]\n")

    return reply


def main():
    print(f"\n🤖 Chatbot ready — Persona: {PERSONA}")
    print("Type 'quit' to exit, 'clear' to reset memory\n")
    print("-" * 50)

    # Initialize with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bye!")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("🧹 Memory cleared.\n")
            continue

        reply = chat(messages, user_input)
        print(f"Bot: {reply}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()