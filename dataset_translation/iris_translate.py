import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "davidkim205/iris-7b"
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(repo)


def read_parquet_file(parquet_file_path):
    df = pd.read_parquet(parquet_file_path)
    return df

def generate(prompt):
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        return_token_type_ids=False
    ).to("cuda")
    gen_tokens = model.generate(
        **encoding,
        max_new_tokens=2048,
        temperature=1.0,
        num_beams=5,
    )
    prompt_end_size = encoding.input_ids.shape[1]
    result = tokenizer.decode(gen_tokens[0, prompt_end_size:])

    return result

def main(parquet_file_path, output_path):
    df = read_parquet_file(parquet_file_path)
    print(df["Description"][41045])
    for d in df:
        print(generate(f"[INST] 다음 문장을 한글로 번역하세요.{d['Description']} [/INST]"))

parquet_file_path = 'dialogues.parquet'  
main(parquet_file_path, "dialogues_kor.parquet")
