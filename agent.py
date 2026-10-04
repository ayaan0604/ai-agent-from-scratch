from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")


client = OpenAI(
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key = gemini_api_key)

model = 'gemini-3.8-flash'

def chat(user_message: str)-> str:
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                'role': "system", 'content' : "You are a helpful personal assistant"
            },
            {
                'role': "user", 'content' : user_message
            }
        ]
    )

    return response.choices[0].message.content or ""

if __name__ == "__main__":
    print(f"v0 assistant ({model}), ctrl+c to quit")
    try:
        while True:
            user_message = input("Enter your message: ")
            print('assistant: ', chat(user_message))
    except(EOFError, KeyboardInterrupt):
        print("bye")