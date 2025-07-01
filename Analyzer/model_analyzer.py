import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import re
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import random
import json
import time

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def setup_gpus(gpu_ids=None):
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu", []
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        available_gpus = list(range(len(gpu_ids)))
        print(f"Using specified GPUs: {gpu_ids}")
    else:
        available_gpus = list(range(torch.cuda.device_count()))
        print(f"Using all available GPUs: {available_gpus}")
    for i, gpu_id in enumerate(available_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
    device = f"cuda:{available_gpus[0]}" if available_gpus else "cpu"
    return device, available_gpus

def get_model_for_generation(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    else:
        return model

def generate_prompt(input_log):
    prompt = f"""You are tasked with analyzing the following log message and explaining your thought process in detail:  

"{input_log}"  

Please follow these steps:  
1. Examine the log message carefully  
2. Identify which parts are likely to be variables (numbers, IDs, timestamps, etc.)  
3. Replace variable parts with "<*>" symbols  
4. PROVIDE DETAILED STEP-BY-STEP REASONING for your decisions  

When responding:  
1. First explain your thought process thoroughly (why you think certain parts are variables)  
2. Then provide the final template with "<*>" replacing variables  
3. Make sure to include "TEMPLATE:" before your final template  

Please be thorough in your analysis and reasoning.
"""
    return prompt

def generate_prompt_list(content_indices, all_logs):
    prompt_list = []
    for log_index in content_indices:
        target_log = all_logs.iloc[log_index]['Content']
        prompt = generate_prompt(target_log)
        prompt_list.append(prompt)
    return prompt_list

def analyze_logs_batch(content_indices, df_logs, model, tokenizer, device, 
                      max_length, max_new_tokens, temperature):

    prompt_list = generate_prompt_list(content_indices, df_logs)
    inputs = tokenizer(
        prompt_list,
        padding=True,
        truncation=True,
        max_length=max_length - max_new_tokens,
        return_tensors="pt"
    )
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": 0.9 if temperature > 0 else None,
        "top_k": 50 if temperature > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    with torch.no_grad():
        try:
            generation_model = get_model_for_generation(model)
            outputs = generation_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs
            )
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory, try reducing batch size...")
            torch.cuda.empty_cache()
            raise

    processed_outputs = []
    input_length = inputs["input_ids"].shape[1]
    for i, output in enumerate(outputs):
        generated_tokens = output[input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        clean_output = generated_text.replace("<unk>", "").replace("</s>", "").replace("<s>", "")
        clean_output = clean_output.replace("<|im_start|>", "").replace("<|im_end|>", "")
        clean_output = clean_output.replace("<|begin_of_text|>", "").replace("<|system|>", "")
        clean_output = clean_output.replace("<|user|>", "").replace("<|assistant|>", "")
        if "TEMPLATE:" not in clean_output and "<*>" in clean_output:
            template_line = None
            for line in clean_output.split("\n"):
                if "<*>" in line:
                    template_line = line.strip()
                    break
            if template_line:
                clean_output += f"\n\nTEMPLATE: {template_line}"
        clean_output = clean_output.strip() + "\n\n\n\n"
        processed_outputs.append(clean_output)
    return processed_outputs

def extract_template(full_output):
    template_match = re.search(r'TEMPLATE:\s*(.*?)(?:\n|\r|$)', full_output, re.IGNORECASE)
    if template_match:
        template = template_match.group(1).strip()
        template = template.strip('"\'')
        return template
    patterns = [
        r'final template:?\s*(.*?)(?:\n|\r|$)',
        r'resulting template:?\s*(.*?)(?:\n|\r|$)',
        r'generated template:?\s*(.*?)(?:\n|\r|$)',
        r'log template:?\s*(.*?)(?:\n|\r|$)',
        r'template:?\s*(.*?)(?:\n|\r|$)',
        r'the template is:?\s*(.*?)(?:\n|\r|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, full_output, re.IGNORECASE)
        if match:
            template = match.group(1).strip()
            template = template.strip('"\'')
            return template
    lines = full_output.split('\n')
    for line in lines:
        line = line.strip()
        if '<*>' in line and 5 < len(line) < 300:
            clean_line = re.sub(r'^(template|the template|here is|this is|output|result|so):?\s*', '', line, flags=re.IGNORECASE)
            clean_line = clean_line.strip('"\'')
            return clean_line
    if lines and lines[0].strip():
        return lines[0].strip()
    return full_output

def initialize_csv_files(output_dir):
    headers = ["Content", "Prediction", "Think-Process", "EventTemplate"]
    prediction_csv = os.path.join(output_dir, "prediction.csv")
    diff_csv = os.path.join(output_dir, "diff.csv")
    same_csv = os.path.join(output_dir, "same.csv")
    for file_path in [prediction_csv, diff_csv, same_csv]:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Reset file: {os.path.basename(file_path)}")
    return prediction_csv, diff_csv, same_csv

def append_to_csv(file_path, rows_data):
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows_data:
            writer.writerow(row)

def process_log_csv(input_csv, output_dir, model, tokenizer, device, batch_size, max_length, max_new_tokens, temperature, sample_size=None):
    os.makedirs(output_dir, exist_ok=True)
    prediction_csv, diff_csv, same_csv = initialize_csv_files(output_dir)
    try:
        print(f"Reading log data from {input_csv}...")
        df = pd.read_csv(input_csv)
        required_cols = ["Content", "EventTemplate"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Input file missing required column '{col}'")
                return {
                    "status": "failed",
                    "error": f"Input file missing required column '{col}'",
                    "processed": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "time_taken": 0
                }
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Randomly sampled {sample_size} records for testing")
        correct_count = 0
        total_count = 0
        start_time = time.time()
        for i in tqdm(range(0, len(df), batch_size), desc="Processing log file"):
            batch_end = min(i + batch_size, len(df))
            batch_indices = list(range(i, batch_end))
            try:
                batch_results = analyze_logs_batch(
                    batch_indices, df, model, tokenizer, device,
                    max_length, max_new_tokens, temperature
                )
            except torch.cuda.OutOfMemoryError:
                print(f"GPU out of memory, skipping batch {i}-{batch_end}")
                continue
            except Exception as e:
                print(f"Error processing batch {i}-{batch_end}: {e}")
                continue
            prediction_rows = []
            same_rows = []
            diff_rows = []
            for j, idx in enumerate(batch_indices):
                if j >= len(batch_results):
                    continue
                total_count += 1
                full_result = batch_results[j]
                predicted_template = extract_template(full_result)
                log_content = df.iloc[idx]['Content']
                correct_template = df.iloc[idx]['EventTemplate']
                csv_row = [log_content, predicted_template, full_result, correct_template]
                prediction_rows.append(csv_row)
                if predicted_template.strip() == correct_template.strip():
                    correct_count += 1
                    same_rows.append(csv_row)
                else:
                    diff_rows.append(csv_row)
            append_to_csv(prediction_csv, prediction_rows)
            if same_rows:
                append_to_csv(same_csv, same_rows)
            if diff_rows:
                append_to_csv(diff_csv, diff_rows)
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                tqdm.write(f"Processed: {total_count}/{len(df)}, Current accuracy: {accuracy:.2f}%")
        time_taken = time.time() - start_time
        final_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        print(f"\nProcessing completed!")
        print(f"Total processed: {total_count} logs, correct predictions: {correct_count}")
        print(f"Accuracy: {final_accuracy:.2f}%")
        print(f"Time taken: {time_taken:.1f} seconds")
        print(f"Results saved to {output_dir}")
        result_summary = {
            "status": "success",
            "processed": total_count,
            "correct": correct_count,
            "accuracy": final_accuracy,
            "time_taken": time_taken,
            "input_file": input_csv,
            "output_dir": output_dir
        }
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, indent=2)
        return result_summary
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Error processing CSV:")
        print(error_details)
        return {
            "status": "failed",
            "error": str(e),
            "error_details": error_details,
            "processed": total_count if 'total_count' in locals() else 0,
            "correct": correct_count if 'correct_count' in locals() else 0,
            "accuracy": (correct_count / total_count * 100) if 'total_count' in locals() and total_count > 0 else 0.0,
            "time_taken": time.time() - start_time if 'start_time' in locals() else 0
        }

def main():
    parser = argparse.ArgumentParser(description='Log Template Analysis Tool (single-file, absolute path)')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model directory")
    parser.add_argument("--input_csv", type=str, required=True, help="Absolute path to your input CSV file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--temperature", type=float, default=0.01, help="Generation temperature")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="List of GPU IDs to use")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of log samples to process")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    args = parser.parse_args()
    print(f"Using model: {args.model_path}")
    print(f"Input CSV file: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All'}")
    device, available_gpus = setup_gpus(args.gpu_ids)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Loading model, please wait...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if len(available_gpus) > 1 else None,
    )
    if len(available_gpus) == 1:
        model = model.to(device)
    elif len(available_gpus) == 0:
        model = model.to("cpu")
    has_device_map = hasattr(model, 'hf_device_map') or hasattr(model, 'device_map')
    if len(available_gpus) > 1 and not has_device_map:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"Using DataParallel on {len(available_gpus)} GPUs")
    elif has_device_map:
        print(f"Using device_map distributed loading on {len(available_gpus)} GPUs")
    model.eval()
    print("Model loaded successfully!")
    process_log_csv(
        args.input_csv,
        args.output_dir,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_length,
        args.max_new_tokens,
        args.temperature,
        args.sample_size
    )

if __name__ == "__main__":
    main()