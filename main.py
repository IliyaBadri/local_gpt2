import asyncio
import json
import logging
from multiprocessing import Process
import os
import random
import time
import torch
import http.server
import socketserver
import websockets
from typing import Callable, List, Optional

def print_exception(exception: Exception):
    raise exception

def print_tokens(tokens: list[str]):
    for token in tokens:
        print(token, end='', flush=True)

def generate_unique_id() -> str:
    timestamp = int(time.time())
    random_digits = random.randint(1000, 9999) 
    unique_id = f"{timestamp}{random_digits}"
    return unique_id

class Model():
    global_model = None
    def __init__(self):
        self.model_name = "openai-community/gpt2"
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        Model.global_model = self

    @staticmethod
    def get_subtensor_from_end(tensor: torch.Tensor, token_count: int) -> torch.Tensor:
        selected_tokens = []
        for i in range(token_count):
            selected_tokens.append(tensor[0, -(token_count - i)]) 
        subtensor = torch.tensor([selected_tokens]) 
        return subtensor

    def initialize_output_tensor(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors="pt")
    
    def get_attention_mask(self, output_tensor: torch.Tensor) -> torch.Tensor:
        attention_mask_array = []
        for sequence in output_tensor:
            mask = []
            for token in sequence:
                if token == self.tokenizer.eos_token_id:
                    mask.append(0)
                else:
                    mask.append(1)

            attention_mask_array.append(mask)

        return torch.tensor(attention_mask_array)
    
    def decode_tokens_tensor(self, tokens_tensor: torch.Tensor) -> List[str]:
        tokens: list[str] = []

        for token_id in tokens_tensor[0]:
            tokens.append(self.tokenizer.decode(token_id, skip_special_tokens=True))
        
        return tokens
    
    def generate_tokens(self, output_tensor: torch.Tensor, token_count: int) -> tuple[Optional[list[str]], Optional[torch.Tensor]]:
        if len(output_tensor[0]) > 0:
            if output_tensor [0, -1] == self.tokenizer.eos_token_id:
                return None, None
        
        attention_mask = self.get_attention_mask(output_tensor)

        generated_tensor = self.model.generate(
            output_tensor,
            max_length = output_tensor.shape[1] + token_count,
            num_return_sequences = 1, 
            no_repeat_ngram_size = 2,
            top_p = 0.92,
            temperature=0.82,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

        new_tokens_tensor = Model.get_subtensor_from_end(generated_tensor, token_count)

        tokens = self.decode_tokens_tensor(new_tokens_tensor)

        return tokens, new_tokens_tensor

class TokenList():
    def __init__(self, tokens: List[str], prompt_id: str):
        self.id = prompt_id,
        self.tokens = tokens

class Prompt():
    def __init__(self, model: Model, prompt: str):
        self.id = generate_unique_id()
        self.model = model
        self.prompt = prompt
        self.output_tensor: torch.Tensor = self.model.initialize_output_tensor(prompt)
    
    def generate_recursively(self, rounds: int, tokens_per_round: int):
        try:
            for _ in range(rounds):
                tokens, new_tokens_tensor = self.model.generate_tokens(self.output_tensor, tokens_per_round)
                if tokens == None or new_tokens_tensor == None:
                    break
                self.output_tensor = torch.cat((self.output_tensor, new_tokens_tensor), dim=1)
                token_list = TokenList(tokens, self.id)
                print_tokens(token_list.tokens) #remove for production
                Pools.add_token_list(token_list)
        except Exception as exception:
            print_exception(exception)

class Pools():
    prompts: List[Prompt] = []
    token_lists: List[TokenList] = []

    @staticmethod
    def add_prompt(prompt: Prompt):
        Pools.prompts.append(prompt)

    @staticmethod
    def add_token_list(token_list: TokenList):
        Pools.token_lists.append(token_list)

    @staticmethod
    def pull_prompt() -> Optional[Prompt]:
        if len(Pools.prompts) <= 0:
            return None
        return Pools.prompts.pop(0)
    
    @staticmethod
    def pull_token_list() -> Optional[TokenList]:
        if len(Pools.token_lists) <= 0:
            return None
        return Pools.token_lists.pop(0)

class StaticServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        static_path = os.path.abspath("./static")
        if self.path == "/":
            self.path = "/static/index.html"
        return super().do_GET()
    
def start_static_server(port=8000):
    with socketserver.TCPServer(("", port), StaticServerHandler) as httpd:
        print(f"[+] Web GUI is up at port http://localhost:{port}/")
        httpd.serve_forever()

async def start_token_generator():
    while True:
        prompt = Pools.pull_prompt()
        if prompt == None:
            await asyncio.sleep(0.1)
            continue

        prompt.generate_recursively(10, 5)

async def handle_websocket_messages(websocket):
    while True:
        message = await websocket.recv()
        try:
            data = json.loads(message)
            prompt_string = data["prompt"]

            if(type(prompt_string) != str):
                print("[-] Invalid JSON received")
                continue
            
            if(Model.global_model == None):
                continue

            prompt = Prompt(Model.global_model, prompt_string)

            print(f"[+] A new prompt has been added to the pool. ( prompt id: {prompt.id} )")

            Pools.add_prompt(prompt)

        except json.JSONDecodeError:
            print("[-] Invalid JSON received")

        await asyncio.sleep(0.1)


async def websocket_handler(websocket):
    print(f"[+] New WebSocket connection: {websocket.remote_address}")
    try:
        token_sender_task = asyncio.create_task(handle_websocket_messages(websocket))
        while True:
            token_list = Pools.pull_token_list()
            if token_list == None:
                await asyncio.sleep(0.1)
                continue
            response = {
                "prompt_id": token_list.id,
                "tokens": token_list.tokens
            }
            await websocket.send(json.dumps(response))
            await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[-] Websocket connection closed: {e}")

    finally:
        token_sender_task.cancel()

async def start_websocket_server(port=8001):
    server = await websockets.serve(websocket_handler, "localhost", port)
    print(f"[+] WebSocket server started at ws://localhost:{port}")
    await server.wait_closed()

async def main_asyncio_coroutine():
    token_generator_task = asyncio.create_task(start_token_generator())
    websocket_server_task = asyncio.create_task(start_websocket_server())

    await asyncio.gather(token_generator_task, websocket_server_task)

if __name__ == "__main__":
    models_path = os.path.abspath("./models")
    os.makedirs(models_path, exist_ok=True)

    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("[+] Loading model . . .")
    model = Model()
    print(f"[+] The {model.model_name} model has been loaded.")

    static_server_process = Process(target=start_static_server)
    static_server_process.start()

    asyncio.run(main_asyncio_coroutine())

    



    