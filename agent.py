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
        f.write(f"-{fact} \n")
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

def chat(messages =[])-> str:

    while True:

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



SYSTEM_PROMPT = '''
    You are a helpful personal assistant.
    You can add and load memeories
    Whenever you find a new fact about the user, save it into the persisten memory.
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