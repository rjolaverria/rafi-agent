import argparse

from agent import Agent

SYSTEM_PROMPT = (
    "You are a helpful computer use agent. You can read and write files, "
    "and execute bash commands to accomplish tasks on the user's behalf."
)


def main():
    parser = argparse.ArgumentParser(description="Run the agent with a message.")
    parser.add_argument("message", help="The initial message to send to the agent")
    parser.add_argument("--model", default="openai/gpt-4.1", help="Model to use")
    args = parser.parse_args()

    agent = Agent(args.model, system_prompt=SYSTEM_PROMPT)
    agent.run([{"role": "user", "content": args.message}])


if __name__ == "__main__":
    main()
