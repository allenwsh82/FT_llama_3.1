# llama2-7b-hf-chat-tune.py
import time
import argparse

from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments)

def main(FLAGS):

    #dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

    #Get your hf_token from huggingface.co webpage
    #Use this hf token if yours can't work:    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
    hf_token = "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"   
    
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"    #Make sure you set the correct path
    
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_auth_token=hf_token)
    
    print('setting training arguments')

    training_arguments = TrainingArguments(
        output_dir="./fine_tuned_llama3.1-8B-Instruct",
        bf16=FLAGS.bf16, #change for CPU
        num_train_epochs=1,
        use_ipex=FLAGS.use_ipex, #change for CPU IPEX
        use_cpu=True,
        fp16_full_eval=False,
    )

    print('Creating SFTTrainer')

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True,
    )

    print('Starting Training')
    start = time.time()

    trainer.train()

    total = time.time() - start
    print(f'Time to tune {total}')    
  
    #save the fine tuned model into a folder
    trainer.save_model("./fine_tuned_llama3.1-8B-Instruct")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-bf16',
                        '--bf16',
                        type=bool,
                        default=True,
                        help="activate mix precision training with bf16")
    parser.add_argument('-ipex',
                        '--use_ipex',
                        type=bool,
                        default=True,
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-msq',
                        '--max_seq_length',
                        type=int,
                        default=8196,
                        help="specifies the number of highest probability tokens to consider at each step")

    FLAGS = parser.parse_args()
    main(FLAGS)
