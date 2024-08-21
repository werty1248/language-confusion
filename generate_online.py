import concurrent.futures
import time
import argparse
import pandas as pd
import requests
import json
import os
from tqdm.auto import tqdm

from gen_utils import load_dataset


def query(messages, args):
    response = requests.post(
      url=args.url,
      headers={"Authorization": f"Bearer {args.api_key}"},
      data=json.dumps({
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "min_p": args.min_p,
      })
    )
    return json.loads(response.text)['choices'][0]['message']['content']

def main(args):
    results = []
    
    dataset = load_dataset(args.test_data_dir, args.task, args.source, args.language)
    print(len(dataset))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for data in dataset:
            messages = []
            if args.system_prompt is not None:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": data['prompt']})
            
            futures.append(executor.submit(query, messages, args))
            
        for idx, future in tqdm(enumerate(futures), total=len(futures)):
            results.append({"id": idx, "model":args.model, "completion":future.result(),
            "task":dataset[idx]['task'], "source":dataset[idx]['source'],"language":dataset[idx]['language']})
    
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output_dir, args.model.split("/")[-1] + ".csv"), encoding="utf-8-sig", index=False)
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--url', type=str)
    parser.add_argument('--api-key', type=str, default='your_api_key')
    
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--model', type=str, default="meta-llama/llama-3.1-8b-instruct")
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--min-p', type=float, default=0.75)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--system-prompt', type=str, default=None)
    
    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--source', type=str, default="all")
    parser.add_argument('--language', type=str, default="all")
    
    parser.add_argument('--test-data-dir', type=str, default="test_sets")
    parser.add_argument('--output-dir', type=str, default="outputs")

    args = parser.parse_args()
    
    print(args)
    
    start = time.time()
    main(args)

    end = time.time()
    print("수행시간: %f 초" % (end - start))