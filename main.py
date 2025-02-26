from flask import Flask, Response, send_from_directory, request, jsonify
import json
import logging
import os
import random
import time
import threading
import torch
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
    def __init__(self):
        self.model_name = "openai-community/gpt2"
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

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
    
    def generate_tokens(self, output_tensor: torch.Tensor, token_count: int) -> Optional[list[str]]:
        if len(output_tensor[0]) > 0:
            if output_tensor [0, -1] == self.tokenizer.eos_token_id:
                return None
        
        attention_mask = self.get_attention_mask(output_tensor)

        generated_tensor = self.model.model.generate(
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

        output_tensor[:] = torch.cat((output_tensor, new_tokens_tensor), dim=1)

        tokens = self.decode_tokens_tensor(new_tokens_tensor)

        return tokens

    

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
                tokens = self.model.generate_tokens(self.output_tensor, tokens_per_round)
                if(tokens == None):
                    break
                token_list = TokenList(tokens, self.id)
                Pools.add_token_list(token_list)
        except Exception as exception:
            print_exception(exception)

class Pools():
    prompts: List[Prompt] = []
    token_lists: List[TokenList] = []

    @staticmethod
    def add_prompt(prompt: Prompt):
        Pools.prompts.append(prompt)

    def add_token_list(token_list: TokenList):
        Pools.token_lists.append(token_list)

def web_server_worker(model: Model, port=8000):
    static_path = os.path.abspath("./static")
    app = Flask(__name__, static_folder=static_path)
    logging.getLogger('werkzeug').disabled = True
    print(f"[+] HTTP server is up ( http://localhost:{port}/index.html )")
    app.run("", port, threaded=True)

if __name__ == "__main__":
    models_path = os.path.abspath("./models")
    os.makedirs(models_path, exist_ok=True)

    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = Model()

    web_server_worker(model, 8000)



    