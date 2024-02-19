import boto3
import cohere
import json
import openai
import os
import torch 

from pathlib import Path 
from typing import List 
from retry import retry 
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self, model, max_tokens=128, quantization=False):
        self.model = model 
        self.max_tokens = max_tokens
        self.quantization = quantization

        self.hf_model_names = {
            "gpt2": "gpt2-large",
            "pythia-2.8B": "EleutherAI/pythia-2.8b",
            "llama2-7B": "meta-llama/Llama-2-7b-hf",
            "llama2-13B": "meta-llama/Llama-2-13b-hf",
            "vicuna-33B": "lmsys/vicuna-33b-v1.3",
            "mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "llama2-70B-chat": "meta-llama/Llama-2-70b-chat-hf",
        }
        self.openai_model_names = {
            "chat": "gpt-3.5-turbo",
            "gpt4": "gpt-4-0613",
            "gpt4.5": "gpt-4-0125-preview"
        }
        if model in self.openai_model_names:
            self.openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.make_query = self.make_chatgpt_query
            
        elif model == "cohere":
            self.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])
            self.make_query = self.make_cohere_query

        elif model == "claude":
            self.bedrock_client = boto3.client(service_name="bedrock-runtime")
            self.make_query = self.make_claude_query
            
        elif model in self.hf_model_names.keys():
            
            model_fullname = self.hf_model_names[model]

            # Applying quantization (8bit or 4bit)
            if quantization == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_fullname,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                )
                self.hf_pipeline = pipeline(
                    task="text-generation", 
                    model=hf_model,
                    tokenizer=AutoTokenizer.from_pretrained(model_fullname),
                    torch_dtype=torch.float16,
                )
            elif quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_fullname,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                )
                self.hf_pipeline = pipeline(
                    task="text-generation", 
                    model=hf_model,
                    tokenizer=AutoTokenizer.from_pretrained(model_fullname),
                    torch_dtype=torch.float16,
                )
            else:
                # No quantization
                if any([s in model for s in ["33B", "8x7B", "56B", "70B"]]):
                    print ('You are using one of the larger models, but the quantization option is not set. Consider quantizatizing it by setting "quantization=4bit" or "8bit" to fit onto the GPU memory.')
                self.hf_pipeline = pipeline(
                    task="text-generation",
                    model=model_fullname,
                    torch_dtype=torch.float16,
                    device=0 if torch.cuda.is_available() else -1,
                    trust_remote_code=True
                )
            
            self.make_query = self.make_huggingface_pipeline_query
        else:    
            raise ValueError("Model {} not supported!".format(model))

    def make_query(self, prompt):
        # To be overritten upon selecting a model at init
        raise NotImplemented()
    

    @retry(tries=2, delay=1)
    def make_chatgpt_query(self, prompt):
        
        resp = self.openai_client.chat.completions.create(
            model=self.openai_model_names[self.model],
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content

    @retry(tries=2, delay=1)
    def make_cohere_query(self, prompt):
        resp = self.cohere_client.generate(
            prompt,
            model="command",
            max_tokens=self.max_tokens
        )
        return resp[0].text 

    def make_claude_query(self, prompt):
        formatted_prompt = f"Human: {prompt} \nAssistant:"
        body = json.dumps({
            "prompt": formatted_prompt,
            "max_tokens_to_sample": self.max_tokens,
        })
        modelId = "anthropic.claude-v2"
        response = self.bedrock_client.invoke_model(
            body=body, modelId=modelId,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        outputText = response_body.get("completion")
        return outputText 

    def make_huggingface_pipeline_query(self, prompt):
        completion = self.hf_pipeline(
            prompt, 
            return_full_text=False,
            max_new_tokens=self.max_tokens, 
            num_return_sequences=1
        )
        return completion[0]["generated_text"]
    

if __name__ == "__main__":
    dummy_prompt = "Tell me a story"

    # model_name = "chat"
    # model_name = "cohere"
    # model_name = "gpt2"
    # model_name = "pythia-2.8B"
    # model_name = "claude"
    model_name = "vicuna-33B"
    # model_name = "llama2-7B"
    llm = LLM(model_name, quantization="4bit")
    s = llm.make_query(dummy_prompt)
    print("{} story: {}".format(model_name, s))
