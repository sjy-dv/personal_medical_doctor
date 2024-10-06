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
    if not text or text.strip() == "":
        return None
    resp = translate_client.translate_text(
        parent=proj,
        contents=[text],
        target_language_code="ko"
    )
    return resp.translations[0].translated_text

def translate_row(row):
    return {
        'question': translate_eng_kor(row['question']),
        'context': translate_eng_kor(row['context']),
    }

def save_partial_data(translate_arr, output_dir, batch_num, parquet_file_name):
    tdf = pd.DataFrame(translate_arr)
    file_base_name = os.path.splitext(os.path.basename(parquet_file_name))[0]
    partial_output_path = f"{output_dir}/{file_base_name}_batch_{batch_num}.parquet"
    tdf.to_parquet(partial_output_path, index=False)
    print(f"Batch {batch_num} saved: {partial_output_path}")

def convert_eng_to_kor_parquet(parquet_file_path: str, output_dir: str):
    translate_arr = []
    df = parquet_reader(parquet_file_path=parquet_file_path)
    total_rows = len(df)
    print(f"Total rows to process in {parquet_file_path}: {total_rows}")
    
    batch_num = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(translate_row, [row for _, row in df.iterrows()]), 
                           total=total_rows, unit=' rows'):
            if result is not None:
                translate_arr.append(result)
            
            if len(translate_arr) >= 50000:
                save_partial_data(translate_arr, output_dir, batch_num, parquet_file_path)
                translate_arr.clear()
                batch_num += 1

    if translate_arr:
        save_partial_data(translate_arr, output_dir, batch_num, parquet_file_path)

    print(f"Translation completed for {parquet_file_path}. All data saved.")

def process_all_parquet_files_in_folder(folder_path: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".parquet"):
            parquet_file_path = os.path.join(folder_path, file_name)
            convert_eng_to_kor_parquet(parquet_file_path, output_dir)

process_all_parquet_files_in_folder("domain_datasets", "custom_domain_datasets")
