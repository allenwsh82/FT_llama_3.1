from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import torch
import transformers
#from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import intel_extension_for_pytorch as ipex
import time
from time import perf_counter
from ipex_llm import optimize_model

#Make sure you download and set the correct path for the Llama-3-8B-Instruct that you downloaded
model_path = "./fine_tuned_llama3.1-8B-Instruct" 
#model_path = "./fine_tuned_llama3.2-3B-Instruct" 

#tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=True
)


###############################################################################################################

# With only one line to enable IPEX-LLM optimization on model to INT4 precision
model = optimize_model(model)

###############################################################################################################

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


streamer = TextStreamer(tokenizer)

print("This Demo is Powered by Intel Xeon on Intel Tiber Developer Cloud")
print()
prompt=input("Please enter your prompt: ")
print()

messages = [
    {"role": "system", "content": "You are a helpful AI assistant"},
    {"role": "user", "content": prompt},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# get the start time
st = time.time()

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    streamer=streamer,
    top_p=0.9,
)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st

response = outputs[0][input_ids.shape[-1]:]
#print(tokenizer.decode(response, skip_special_tokens=True))

print('Execution time:', elapsed_time, 'seconds')
print()
print("This is running inference with Xeon CPU with ipex-llm (INT4)")
print()
token_num = len(outputs[0])
print('-'*52)
print('Number of tokens:', token_num)
print(f'Inference time: {et-st} s')
print(f'Token/s: {token_num/(et-st)}')
print('-'*20, 'Outputs', '-'*20)
