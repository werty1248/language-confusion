from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import argparse
import pandas as pd
import os

from gen_utils import load_dataset

def chat_template(tokenizer, data, args):
    messages = []
    if args.system_prompt is not None:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": data['prompt']})
    
    return tokenizer.apply_chat_template(messages ,tokenize=False, add_generation_prompt=True)

def main(args):
    results = []

    if args.hf_token is not None:
        from huggingface_hub import login
        login(token = args.hf_token)

    dataset = load_dataset(args.test_data_dir, args.task, args.source, args.language)
    print(len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    prompts = [chat_template(tokenizer, data, args) for data in dataset]
    
    sampling_params = SamplingParams(temperature=args.temperature, min_p=args.min_p, max_tokens=args.max_tokens,)

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.95, dtype=args.dtype)
    outputs = llm.generate(prompts, sampling_params)
    
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({"id": idx, "model":args.model, "completion":generated_text,
                        "task":dataset[idx]['task'], "source":dataset[idx]['source'],"language":dataset[idx]['language']})
                        
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output_dir, args.model.split("/")[-1] + ".csv"), encoding="utf-8-sig", index=False)
    
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hf_token', type=str, default=None)
    
    parser.add_argument('--model', type=str, default="meta-llama/meta-llama-3.1-8b-instruct")
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--min_p', type=float, default=0.75)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--system_prompt', type=str, default=None)
    
    parser.add_argument('--dtype', type=str, default='auto')
    parser.add_argument('--max_model_len', type=int, default=4096)
    
    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--source', type=str, default="all")
    parser.add_argument('--language', type=str, default="all")
    
    parser.add_argument('--test_data_dir', type=str, default="test_sets")
    parser.add_argument('--output_dir', type=str, default="outputs")

    args = parser.parse_args()
    
    print(args)
    
    start = time.time()
    main(args)

    end = time.time()
    print("수행시간: %f 초" % (end - start))