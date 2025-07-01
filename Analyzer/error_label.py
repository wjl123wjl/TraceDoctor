import os
import csv
import pandas as pd
from tqdm import tqdm
import time
import logging
import re
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepSeekLogClassifier:
    def __init__(self, api_key=None, model_name="deepseek-chat", md_summary_path=None, batch_size=1):
        self.batch_size = batch_size
        self.model_name = model_name
        self.md_summary_path = md_summary_path

        # Get API Key
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("DeepSeek API key missing! Provide via param or env DEEPSEEK_API_KEY")

        # API endpoint
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"

        logger.info(f"Initialized DeepSeek API (model: {self.model_name})")

        # Load error type summary and mapping from MD file
        self.error_types_summary, self.error_type_mapping = self._load_error_types_from_md()
        
        if not self.error_types_summary:
            logger.error("Failed to load error types from MD file. Cannot proceed without error type definitions.")
            raise ValueError("MD summary file is required and must contain valid error type definitions")

    def _load_error_types_from_md(self):
        """
        Load error type summary from MD file and build mapping
        """
        if not self.md_summary_path or not os.path.exists(self.md_summary_path):
            logger.error(f"MD summary file not found: {self.md_summary_path}")
            return None, {}
        
        with open(self.md_summary_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Parse error types
        error_type_mapping = self._parse_error_types_from_content(md_content)
        
        if not error_type_mapping:
            logger.error("No valid error types found in MD file")
            return None, {}
        
        logger.info(f"Successfully loaded {len(error_type_mapping)} error types from: {self.md_summary_path}")
        logger.info(f"Error types: {list(error_type_mapping.values())}")
        
        return md_content, error_type_mapping

    def _parse_error_types_from_content(self, md_content):
        """
        Parse error types from MD content, supporting letter identifiers (A, B, C, etc.)
        """
        error_type_mapping = {}
        
        # Split content by lines
        lines = md_content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for format like "#### A. Path Variable Fragmentation"
            if line.startswith('####') and '. ' in line:
                parts = line.replace('####', '').strip().split('. ', 1)
                if len(parts) == 2:
                    letter_part = parts[0].strip()
                    title_part = parts[1].strip()
                    
                    # Check if it's a single letter
                    if len(letter_part) == 1 and letter_part.isalpha():
                        number = ord(letter_part.upper()) - ord('A') + 1
                        error_type_mapping[number] = title_part
        
        return error_type_mapping

    def _create_classification_prompt(self, original_log, reasoning_process, correct_template):
        """
        Create classification prompt, directly accepting parameters without template formatting
        """
        # Build error type options list
        options_text = ""
        for num, error_type in self.error_type_mapping.items():
            letter = chr(ord('A') + num - 1)  # Convert to letter (1->A, 2->B, etc.)
            options_text += f"{num} ({letter}): {error_type}\n"
        
        prompt = f"""You are a log analysis tool. Classify the log parsing error into one of these categories:

{options_text}

Analyze the following:
Original Log: {original_log}
Model's Reasoning: {reasoning_process}
Correct Template: {correct_template}

Output ONLY a JSON with two fields:
- "label": number from 1 to {len(self.error_type_mapping)}
- "error_type": the category name

Do not include any explanations or additional text."""

        return prompt

    def _call_deepseek_api(self, messages, temperature=0.7):
        """
        Call the DeepSeek API
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000,
            "top_p": 0.95,
            "stream": False
        }

        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None

    def classify_batch_logs(self, log_entries, output_file):
        """
        Batch classify log error types
        """
        results = []
        total_count = len(log_entries)
        start_time = time.time()
        
        logger.info(f"Starting to process {total_count} log records")
        
        for i, entry in enumerate(tqdm(log_entries, desc="Classifying log error types")):
            original_log = entry.get('original_log', '')
            reasoning_process = entry.get('reasoning_process', '')
            correct_template = entry.get('correct_template', '')
            
            # Skip empty fields
            if not original_log or not reasoning_process or not correct_template:
                logger.warning(f"Skipping record {i+1} due to empty fields")
                continue
            
            # Get classification result
            classification_result = self._classify_single_log(original_log, reasoning_process, correct_template)
            
            # Add to results list
            result_entry = {
                'Original_Log': original_log,
                'Correct_Template': correct_template,
                'Error_Type': classification_result['error_type'],
                'Error_Number': classification_result['label']
            }
            results.append(result_entry)
            
            logger.info(f"Record {i+1}: Error type {result_entry['Error_Number']} - {result_entry['Error_Type']}")
            
            # Periodically save results
            if (i + 1) % self.batch_size == 0 or (i + 1) == total_count:
                self._save_results(results, output_file)
                
                # Update progress information
                elapsed_time = time.time() - start_time
                logs_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                estimated_remaining = (total_count - (i + 1)) / logs_per_second if logs_per_second > 0 else 0
                
                logger.info(f"Progress: {i+1}/{total_count} ({(i+1)/total_count*100:.1f}%) - "
                           f"Speed: {logs_per_second:.2f} logs/sec - "
                           f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        # Final save
        self._save_results(results, output_file)
        
        total_time = time.time() - start_time
        logger.info(f"Log classification completed. Processed {len(results)} logs, total time: {total_time/60:.1f} minutes")
        
        return output_file

    def _classify_single_log(self, original_log, reasoning_process, correct_template):
        """
        Classify error type for a single log
        """
        # Create prompt messages
        user_content = self._create_classification_prompt(original_log, reasoning_process, correct_template)
        
        messages = [
            {"role": "system", "content": "You are a professional log analysis tool."},
            {"role": "user", "content": user_content}
        ]
        
        # Get model response
        response_data = self._call_deepseek_api(messages)
        
        if not response_data:
            return {"label": 0, "error_type": "API_Error"}
        
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            return {"label": 0, "error_type": "No_Response"}
        
        response_text = response_data['choices'][0]['message']['content'].strip()
        
        if not response_text:
            return {"label": 0, "error_type": "Empty_Response"}
        
        # Parse JSON response
        parsed_result = self._parse_json_response(response_text)
        return parsed_result

    def _parse_json_response(self, response_text):
        """
        Parse JSON response
        """
        # Clean response text
        response_text = response_text.strip()
        
        # Find JSON part
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(response_text):
            if char == '{':
                json_start = i
                break
        
        for i in range(len(response_text) - 1, -1, -1):
            if response_text[i] == '}':
                json_end = i + 1
                break
        
        if json_start == -1 or json_end == -1:
            return {"label": 0, "error_type": "No_JSON_Found"}
        
        json_text = response_text[json_start:json_end]
        
        # Parse JSON
        parsed_json = None
        if json_text:
            parsed_json = json.loads(json_text)
        
        if not parsed_json:
            return {"label": 0, "error_type": "JSON_Parse_Error"}
        
        # Extract label and error_type
        label = parsed_json.get('label', 0)
        error_type = parsed_json.get('error_type', 'Unknown')
        
        # Validate label
        if label not in self.error_type_mapping:
            return {"label": 0, "error_type": "Invalid_Label"}
        
        # If error_type is empty, get from mapping
        if not error_type or error_type == 'Unknown':
            error_type = self.error_type_mapping.get(label, 'Unknown')
        
        return {"label": label, "error_type": error_type}

    def _save_results(self, results, output_file):
        """
        Save results to CSV file
        """
        if not results:
            logger.warning("No results to save")
            return
            
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.debug(f"Results saved to: {output_file}")

    def classify_from_csv(self, input_file, output_file):
        """
        Read data from CSV file and classify
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return
        
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} log records from {input_file}")
        
        # Check required columns
        required_columns = ['Content', 'Think-Process', 'EventTemplate']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Input file missing required columns. Required: {required_columns}, Found: {df.columns.tolist()}")
            return
        
        # Convert to log entries list
        log_entries = []
        for _, row in df.iterrows():
            if pd.notna(row['Content']) and pd.notna(row['Think-Process']) and pd.notna(row['EventTemplate']):
                log_entries.append({
                    'original_log': str(row['Content']),
                    'reasoning_process': str(row['Think-Process']),
                    'correct_template': str(row['EventTemplate'])
                })
        
        if not log_entries:
            logger.error("No valid log entries to process")
            return
        
        logger.info(f"Ready to process {len(log_entries)} valid log records")
        
        # Execute classification
        return self.classify_batch_logs(log_entries, output_file)

    def get_error_type_summary(self):
        """
        Return error type summary information
        """
        summary = "Error Type Mapping:\n"
        for num, error_type in self.error_type_mapping.items():
            letter = chr(ord('A') + num - 1)
            summary += f"{num} ({letter}): {error_type}\n"
        return summary


def main():
    parser = argparse.ArgumentParser(description='Classify log parsing errors using DeepSeek API')
    parser.add_argument('--api_key', type=str, help='DeepSeek API key')
    parser.add_argument('--model', type=str, default="deepseek-chat", help='DeepSeek model name')
    parser.add_argument('--md_summary_path', type=str, required=True, help='Path to MD file containing error type summaries')
    parser.add_argument('--input_file', type=str, help='Input CSV file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = DeepSeekLogClassifier(
        api_key=args.api_key,
        model_name=args.model,
        md_summary_path=args.md_summary_path,
        batch_size=args.batch_size
    )
    
    # Display loaded error types
    logger.info("Loaded error types:")
    logger.info(classifier.get_error_type_summary())
    
    # Read from CSV file and classify
    if args.input_file:
        classifier.classify_from_csv(args.input_file, args.output_file)
    else:
        logger.error("Please provide input file path")


if __name__ == "__main__":
    main()