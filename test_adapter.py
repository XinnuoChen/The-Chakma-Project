import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = r"C:\Users\Ed\OneDrive\Desktop\The-Chakma-Project\llama31_dict_lora\adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a concise bilingual dictionary assistant. "
            "Given a headword and optional part of speech, return only the English gloss. "
            "Preserve semicolon-separated meanings exactly if present. "
            "Do not add commentary, examples, or extra wording."
        ),
    },
    {
        "role": "user",
        "content": "What does 'Kushāsan' mean in English?\nPart of speech: n",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = inputs["input_ids"].shape[1]
answer = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
print(answer)