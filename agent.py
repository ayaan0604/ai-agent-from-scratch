from openai import OpenAI
import dotenv
import os
from pathlib import Path
import json

dotenv.load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

WORKSPACE = Path(__file__).parent / 'workspace'




client = OpenAI(
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key = gemini_api_key)

model = 'gemini-3.5-flash-lite'


def list_files()-> str:
    '''list files in the workspace dir'''
    return '\n'.join(p.name for p in WORKSPACE.iterdir()) or 'empty'

def read_file(file_name: str)-> str:
    '''read a file from the workspace'''

    path = WORKSPACE / file_name
    if not path.is_file():
        return f"Error: no file named {file_name}"

    return path.read_text(encoding='utf-8')



TOOLS = {
    'list_files' : list_files,
    'read_file' : read_file
}

TOOLS_SCHEMAS = [
    {
        'type' : "function",
        'function' : {
            'name' : 'list_files',
            'description' : "list the files in the users workspace folder" ,
            'parameters' : {
                'type' : "object",
                'properties' : {}
            }
        }
    },

    {
        'type' : "function",
        'function' : {
            'name' : 'read_file',
            'description' : "read one file from the user's workspace directory" ,
            'parameters' : {
                'type' : 'object',
                'properties' : {
                    'file_name' : {'type' : "string"}
                },
                'required' : ['file_name']
            }
        }
    }
]

def chat(user_message: str)-> str:

    messages = [
            { 'role': "system", 'content' : "You are a helpful personal assistant" },
            {'role': "user", 'content' : user_message }
        ]

    response = client.chat.completions.create(
        model = model,
        messages = messages,
        tools= TOOLS_SCHEMAS
    )

    message =  response.choices[0].message
    if not message.tool_calls:
        return message.content or ""

    messages.append(message)
    
    for call in message.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        print(f"[TOOL] : {call.function.name}({args})")

        result = TOOLS[call.function.name](**args)

        messages.append(
            {'role' : 'tool', 'tool_call_id': call.id, 'content': result}
        )

    final = client.chat.completions.create(
        model = model, messages = messages
    )

    return final.choices[0].message.content or ""

if __name__ == "__main__":
    print(f"v0 assistant ({model}), ctrl+c to quit")
    try:
        while True:
            user_message = input("Enter your message: ")
            print('assistant: ', chat(user_message))
    except(EOFError, KeyboardInterrupt):
        print("bye")