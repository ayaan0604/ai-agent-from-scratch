from openai import OpenAI
import dotenv
import os
from pathlib import Path
import json



dotenv.load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

WORKSPACE = Path(__file__).parent / 'workspace'
MEMORY_FILE = Path(__file__).parent/ "memory.md"



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

def write_file(file_name: str, content: str)->str:
    '''write or overwrite a file in workdir'''

    (WORKSPACE / file_name).write_text(content, encoding='utf-8')

    return f"wrote {file_name}"
        
def read_memory()-> str:
    '''read all the facts stored'''
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text()
    return "nothing saved yet"

def save_memory(fact: str)-> str:
    '''save a fact about the user in the memory'''
    with MEMORY_FILE.open("a", encoding='utf-8') as f:
        f.write(f"- {fact} \n")
    return f'saevd: {fact}'
    


TOOLS = {
    'list_files' : list_files,
    'read_file' : read_file,
    'write_file' : write_file,
    'read_memory' : read_memory,
    'save_memory' : save_memory
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
    },

    {
        'type' : "function",
        'function' : {
            'name' : 'write_file',
            'description' : "write or overwrite user's workspace directory" ,
            'parameters' : {
                'type' : 'object',
                'properties' : {
                    'file_name' : {'type' : "string"},
                    'content' : {'type' :'string'}
                },
                'required' : ['file_name', 'content' ]
            }
        }
    },

    {
        'type' : "function",
        'function' : {
            'name' : 'save_memory',
            'description' : "save a fact about the user in the memory" ,
            'parameters' : {
                'type' : 'object',
                'properties' : {
                    'fact' : {'type' : "string"}
                },
                'required' : ['fact' ]
            }
        }
    },

    {
        'type' : "function",
        'function' : {
            'name' : 'read_memory',
            'description' : "add a fact in the persistent memory that can be retrieved later" ,
            'parameters' : {
                'type' : 'object',
                'properties' : {}
            }
        }
    }
]

def chat(messages =[], max_iter = 10)-> str:

    for _ in range(max_iter):

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

            tool = TOOLS.get(call.function.name)
            if not tool:
                result = f"Error: Unknown tool : {call.function.name}"

            else:
                try: 
                    result = tool(**args)
                except Exception as e:
                    result = f"toll error: {e}"

            messages.append(
                {'role' : 'tool', 'tool_call_id': call.id, 'content': result}
            )



SYSTEM_PROMPT = '''
You are a helpful personal assistant with access to a user's workspace
and persistent memory.

You can:
- inspect files in the workspace
- read files
- create or modify files
- read persistent memories
- save useful facts about the user

Use tools whenever they are necessary to fulfill the user's request.

MEMORY RULES:
- Save information only when it is a useful, reasonably stable fact
  about the user.
- Do not save temporary statements, casual conversation, or task-specific
  information unless the user explicitly asks you to remember it.
- If the user explicitly says "remember this", save it.
- Before answering questions about what you remember about the user,
  use read_memory.

WORKSPACE RULES:
- Never claim to have inspected a file unless you actually used read_file.
- Never claim to have created or modified a file unless you actually
  used write_file.
- You may call multiple tools when a task requires multiple steps.
- After receiving tool results, decide whether another tool is needed
  or whether you can answer the user.
'''

if __name__ == "__main__":
    print(f"v0 assistant ({model}), ctrl+c to quit")
    messages = [
        {'role': 'system', 'content':SYSTEM_PROMPT}
    ]
    try:
        while True:
            user_message = input("Enter your message: ")
            messages.append({'role': "user", 'content' : user_message})
            print('assistant: ', chat(messages))
    except(EOFError, KeyboardInterrupt):
        print("bye")