import os
import pandas as pd
from google.cloud import translate
import concurrent.futures
from tqdm import tqdm 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
proj = ""
translate_client = translate.TranslationServiceClient()

def parquet_reader(parquet_file_path: str):
    df = pd.read_parquet(parquet_file_path)
    return df

def translate_eng_kor(text: str) -> str: 
    resp = translate_client.translate_text(
        parent=proj,
        contents=[text],
        target_language_code="ko"
    )
    return resp.translations[0].translated_text

def translate_row(row):
    return {
        'Description': translate_eng_kor(row['Description']),
        'Patient': translate_eng_kor(row['Patient']),
        'Doctor': translate_eng_kor(row['Doctor'])
    }

def save_partial_data(translate_arr, output_dir, batch_num):
    tdf = pd.DataFrame(translate_arr)
    partial_output_path = f"{output_dir}_batch_{batch_num}.parquet"
    tdf.to_parquet(partial_output_path, index=False)
    print(f"Batch {batch_num} saved: {partial_output_path}")

def convert_eng_to_kor_parquet(parquet_file_path: str, output_dir: str):
    translate_arr = []
    df = parquet_reader(parquet_file_path=parquet_file_path)
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    
    batch_num = 1 

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(translate_row, [row for _, row in df.iterrows()]), 
                           total=total_rows, unit=' rows'):
            translate_arr.append(result)
            
            if len(translate_arr) >= 50000:
                save_partial_data(translate_arr, output_dir, batch_num)
                translate_arr.clear() 
                batch_num += 1

    if translate_arr:
        save_partial_data(translate_arr, output_dir, batch_num)

    print(f"Translation completed. All data saved.")

# 함수 실행
convert_eng_to_kor_parquet("dialogues.parquet", "dialoges_kor.parquet")
