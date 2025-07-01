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
import random

# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

class DeepSeekLogAugmentor:  
    def __init__(self, api_key=None, model_name="deepseek-chat", batch_size=5, error_types_file=None):  
        self.batch_size = batch_size  # Number of logs to process in each batch  
        self.model_name = model_name  # DeepSeek API model name  
        self.augmentation_count = 10  # Number of augmentations per log  
        
        # Get API key (from parameter or environment variable)  
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")  
        if not self.api_key:  
            logger.warning("No DeepSeek API key provided. Please provide it via parameter or DEEPSEEK_API_KEY environment variable")  
        
        # API endpoint  
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"  
        
        logger.info(f"Initializing DeepSeek API (model: {model_name}) for log augmentation")  
        
        # Load error types definition
        self.error_types_content = self._load_error_types(error_types_file)
        
        # Base instruction template
        self.base_instruction = "Convert the following log into a standardized template by identifying and replacing the variable parts with a <*>:"
        
        # Method 1: Variable Substitution
        self.augmentation_template_1 = """
You are a professional log augmentation tool specializing in variable substitution while preserving log semantics.

Given the following information:
- Original Log: The source log message
- Template: The correct template with <*> marking variable parts  
- Error Explanation: Description of why the original parsing failed
- Error Type: The high-level category of the parsing error

Your task is to generate {count} new log variants by:
1. Only replacing parts that correspond to <*> in the template
2. Generating semantically reasonable values for variable parts (numbers, timestamps, IDs, etc.)
3. Keeping all constant parts exactly the same
4. Ensuring the generated logs would still match the same template

The error explanation helps you understand what went wrong in the original parsing, so you can generate similar cases that might face the same parsing challenges.

Output only a JSON array of strings, where each string is a new log variant.
No explanations, just the JSON array.
"""

        # Method 2: Constant-inclusive Rewriting 
        self.augmentation_template_2 = """
You are a professional log augmentation tool specializing in constant-inclusive rewriting.

Given the following information:
- Original Log: The source log message
- Template: The original template with <*> marking variable parts
- Error Explanation: Description of why the original parsing failed  
- Error Type: The high-level category of the parsing error

Your task is to generate {count} new log variants by:
1. Modifying constant parts of the log with semantically equivalent expressions
2. Optionally changing variable parts as well
3. Maintaining the overall semantic meaning and log structure
4. Creating variations that preserve the parsing challenge described in the error explanation

Since you're changing constant parts, you must also generate the correct template for each new log.

Output format: JSON array of objects, each containing:
- "example": The new log message
- "template": The correct template for this new log (with <*> for variable parts)

No explanations, just the JSON array.
"""

        # Method 3: Semantic Rewriting (will be completed in _create_augmentation_messages)
        self.augmentation_template_3_base = """
You are a professional log augmentation tool specializing in semantic rewriting to preserve or induce specific parsing errors.

You will perform one of two tasks based on the input:

TASK A - Error-Preserving Rewriting:
When given a log that already triggers a specific error type, generate new logs that would trigger the same type of error.

TASK B - Error-Inducing Rewriting:  
When given any log, rewrite it to trigger a specific target error type.

{error_types_reference}

Your task is to generate {count} new examples that demonstrate the same error type.

Output format: JSON array of objects, each containing:
- "example": The new log message  
- "template": The template that demonstrates the error type
- "error_type": The error type number (1-6)

No explanations, just the JSON array.
"""

    def _load_error_types(self, error_types_file):
        """
        Load error types definition from markdown file
        """
        if not error_types_file:
            logger.error("Error types file path is required for method 3 augmentation")
            raise ValueError("error_types_file parameter is required")
            
        if not os.path.exists(error_types_file):
            logger.error(f"Error types file not found: {error_types_file}")
            raise FileNotFoundError(f"Error types file not found: {error_types_file}")
            
        try:
            with open(error_types_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully loaded error types definition from {error_types_file}")
            return content
        except Exception as e:
            logger.error(f"Error loading error types file {error_types_file}: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))  
    def _call_deepseek_api(self, messages, temperature=0.7):  
        """  
        Call the DeepSeek API and handle possible errors  
        """  
        headers = {  
            "Content-Type": "application/json",  
            "Authorization": f"Bearer {self.api_key}"  
        }  
        
        payload = {  
            "model": self.model_name,  
            "messages": messages,  
            "temperature": temperature,  
            "max_tokens": 2000,  
            "top_p": 0.95,  
            "stream": False  
        }  
        
        try:  
            response = requests.post(self.api_endpoint, json=payload, headers=headers)  
            response.raise_for_status()  # Raise exception if status code is not 200  
            return response.json()  
        except requests.exceptions.RequestException as e:  
            logger.error(f"API request error: {e}")  
            if hasattr(response, 'status_code'):
                if response.status_code == 429:  
                    logger.warning("Rate limit reached, retrying...")  
                    time.sleep(10)  # Force wait to avoid continuous rate limiting  
                elif response.status_code == 500:  
                    logger.error("Server error, retrying...")  
            raise  # Re-raise the exception for the retry decorator to handle  

    def augment_logs(self, input_file, output_file=None):  
        """  
        Read logs from CSV file and generate augmented versions in JSON format
        """  
        # Check input file  
        if not os.path.exists(input_file):  
            logger.error(f"Input file not found: {input_file}")  
            return  
        
        # If output file not specified, create in the same directory  
        if output_file is None:  
            input_dir = os.path.dirname(input_file)  
            input_basename = os.path.basename(input_file)  
            output_basename = f"augmented_{os.path.splitext(input_basename)[0]}.json"  
            output_file = os.path.join(input_dir, output_basename)  
        
        # Read input CSV file  
        try:  
            df = pd.read_csv(input_file)  
            logger.info(f"Loaded {len(df)} log entries from {input_file}")  
        except Exception as e:  
            logger.error(f"Error reading input file: {e}")  
            return  
        
        # Ensure required columns exist  
        required_columns = ['Original_Log', 'Correct_Template', 'error_summary', 'Error_Type', 'Error_Number']  
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:  
            logger.error(f"Input file missing required columns. Required: {required_columns}, Found: {df.columns.tolist()}")  
            return  
        
        # Create output JSON list - start with original data  
        output_json_list = []
        
        # Add original data to output
        for _, row in df.iterrows():
            original_entry = {
                "instruction": self.base_instruction,
                "input": str(row['Original_Log']),
                "output": str(row['Correct_Template'])
            }
            output_json_list.append(original_entry)
        
        logger.info(f"Added {len(output_json_list)} original log entries to output")  
        
        # Progress tracking  
        processed_count = 0  
        total_count = len(df)  
        start_time = time.time()  
        
        # Process each log  
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting logs"):  
            try:  
                original_log = row['Original_Log']  
                correct_template = row['Correct_Template']  
                error_summary = row['error_summary']
                error_type = row['Error_Type']
                error_number = row['Error_Number']
                
                # Skip empty fields  
                if pd.isna(original_log) or pd.isna(correct_template):  
                    logger.warning(f"Skipping record due to empty fields")  
                    continue  
                
                # Get augmented logs  
                augmented_results = self._augment_single_log(
                    original_log, correct_template, error_summary, error_type, error_number
                )  
                
                # Add augmented logs to output JSON list  
                for result in augmented_results:  
                    augmented_entry = {
                        "instruction": self.base_instruction,
                        "input": result["input"],
                        "output": result["output"]
                    }
                    output_json_list.append(augmented_entry)
                
                processed_count += 1  
                
                # Periodically save results  
                if processed_count % self.batch_size == 0 or processed_count == total_count:  
                    # Save current results  
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_json_list, f, ensure_ascii=False, indent=2)
                    
                    # Update progress information  
                    elapsed_time = time.time() - start_time  
                    logs_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0  
                    estimated_remaining = (total_count - processed_count) / logs_per_second if logs_per_second > 0 else 0  
                    
                    logger.info(f"Progress: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%) - "  
                               f"Speed: {logs_per_second:.2f} logs/sec - "  
                               f"Est. time remaining: {estimated_remaining/60:.1f} minutes")  
                
            except Exception as e:  
                logger.error(f"Error processing log: {e}")  
                import traceback  
                logger.error(traceback.format_exc())  
                continue  
        
        # Final save  
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_json_list, f, ensure_ascii=False, indent=2)
        
        total_time = time.time() - start_time  
        total_augmented = len(output_json_list) - len(df)  # Subtract original count to get only augmented count  
        logger.info(f"Log augmentation complete. Original logs: {len(df)}, Augmented: {total_augmented}, "  
                   f"Total entries: {len(output_json_list)}, Total time: {total_time/60:.1f} minutes")  
        
        return output_file  

    def _augment_single_log(self, original_log, template, error_summary, error_type, error_number):  
        """  
        Augment a single log to generate multiple variants using the three methods
        """  
        # Randomly select augmentation method
        augmentation_method = random.randint(1, 3)
        logger.debug(f"Using augmentation method {augmentation_method} for log")
        
        # Create prompt messages based on selected method
        messages = self._create_augmentation_messages(
            original_log, template, error_summary, error_type, error_number, method=augmentation_method
        )
        
        # Get model response  
        start_time = time.time()  
        logger.debug(f"Starting log augmentation: {original_log[:50]}...")  
        response_text = self._get_model_response(messages)  
        inference_time = time.time() - start_time  
        logger.debug(f"Augmentation complete, time taken: {inference_time:.2f} seconds")  
        
        # Parse response based on the augmentation method used
        if augmentation_method == 1:
            augmented_results = self._parse_augmentation_response_method1(response_text, original_log, template)
        elif augmentation_method == 2:
            augmented_results = self._parse_augmentation_response_method2(response_text, original_log, template)
        else:  # method 3
            augmented_results = self._parse_augmentation_response_method3(response_text, original_log, template)
        
        # Ensure we have enough augmented logs  
        while len(augmented_results) < self.augmentation_count:  
            logger.warning(f"Insufficient augmented logs ({len(augmented_results)}/{self.augmentation_count}), attempting to generate more")  
            # Use the same method for additional augmentations
            additional_results = self._get_additional_augmentations(
                original_log, template, error_summary, error_type, error_number,
                self.augmentation_count - len(augmented_results),  
                augmented_results,
                method=augmentation_method
            )  
            augmented_results.extend(additional_results)  
            # Avoid infinite loop  
            if not additional_results:  
                break  
        
        # Take first N augmented logs  
        return augmented_results[:self.augmentation_count]  

    def _create_augmentation_messages(self, original_log, template, error_summary, error_type, error_number, method=1):  
        """  
        Create messages format for log augmentation based on the selected method
        """  
        # System message  
        system_message = {"role": "system", "content": "You are a professional log augmentation tool that generates semantically meaningful log variants for training parsing models."}  
        
        # Select template based on method
        if method == 1:
            user_content = self.augmentation_template_1.format(count=self.augmentation_count)
            user_content += f"\n\nOriginal Log: {original_log}\n"
            user_content += f"Template: {template}\n"
            user_content += f"Error Explanation: {error_summary}\n"
            user_content += f"Error Type: {error_type}"
            
        elif method == 2:
            user_content = self.augmentation_template_2.format(count=self.augmentation_count)
            user_content += f"\n\nOriginal Log: {original_log}\n"
            user_content += f"Template: {template}\n"
            user_content += f"Error Explanation: {error_summary}\n"
            user_content += f"Error Type: {error_type}"
            
        else:  # method 3
            # Insert error types reference into template
            user_content = self.augmentation_template_3_base.format(
                count=self.augmentation_count,
                error_types_reference=self.error_types_content
            )
            user_content += f"\n\nOriginal Log: {original_log}\n"
            user_content += f"Template: {template}\n"
            user_content += f"Error Explanation: {error_summary}\n"
            user_content += f"Error Type: {error_type}\n"
            user_content += f"Error Number: {error_number}\n\n"
            user_content += "Task: Generate new examples that would trigger the same type of parsing error."
        
        user_message = {"role": "user", "content": user_content}  
        
        return [system_message, user_message]  

    def _get_model_response(self, messages):  
        """  
        Get model response via DeepSeek API  
        """  
        try:  
            # Call API  
            response_json = self._call_deepseek_api(messages, temperature=0.7)  
            
            # Extract generated text  
            if 'choices' in response_json and len(response_json['choices']) > 0:  
                response_text = response_json['choices'][0]['message']['content']  
                return response_text  
            else:  
                logger.error(f"Abnormal API response format: {response_json}")  
                return ""  
        except Exception as e:  
            logger.error(f"Error getting model response: {e}")  
            return ""  

    def _parse_augmentation_response_method1(self, response_text, original_log, template):  
        """  
        Parse the augmented log response from API for method 1 (Variable Substitution)
        Method 1 only changes variables, so template remains the same
        """  
        try:  
            # Clean response text, extract JSON part  
            response_text = response_text.strip()  
            
            # Try to find JSON array part  
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)  
            if json_match:  
                json_str = json_match.group(0)  
            else:  
                # If no clear JSON format, try splitting by lines  
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]  
                # Filter out lines that are obviously not logs  
                potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]  
                return [{"input": log, "output": template} for log in potential_logs]  
            
            # Parse JSON  
            augmented_logs = json.loads(json_str)  
            
            # Ensure result is a list of strings  
            if not isinstance(augmented_logs, list):  
                logger.warning(f"API returned non-list format: {type(augmented_logs)}")  
                return []  
            
            # Filter and clean augmented logs  
            valid_results = []  
            for log in augmented_logs:  
                if not isinstance(log, str):  
                    continue  
                    
                # Clean quotes and whitespace in log  
                clean_log = log.strip().strip('"\'')  
                
                # Skip augmentations identical to original log  
                if clean_log == original_log:  
                    continue  
                    
                valid_results.append({"input": clean_log, "output": template})  
            
            return valid_results  
            
        except json.JSONDecodeError:  
            logger.warning(f"JSON parsing failed, attempting fallback parsing method")  
            # Fallback method: split response by line  
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]  
            # Filter out lines that are obviously not logs (like code block markers)  
            potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]  
            return [{"input": log, "output": template} for log in potential_logs]
        except Exception as e:  
            logger.error(f"Error parsing augmentation response: {e}")  
            return []  

    def _parse_augmentation_response_method2(self, response_text, original_log, template):
        """
        Parse the augmented log response from API for method 2 (Constant-inclusive Rewriting)
        Method 2 changes constants, so new templates are needed
        """
        try:
            # Clean response text, extract JSON part
            response_text = response_text.strip()
            
            # Try to find JSON array part
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                
                # Extract examples and templates from JSON objects
                valid_results = []
                for item in json_data:
                    if isinstance(item, dict) and "example" in item and "template" in item:
                        example = item["example"].strip()
                        new_template = item["template"].strip()
                        # Skip if identical to original
                        if example != original_log:
                            valid_results.append({"input": example, "output": new_template})
                
                return valid_results
            else:
                # No JSON array found, try to extract JSON objects
                json_objects = re.findall(r'\{[^{}]*"example"[^{}]*"template"[^{}]*\}', response_text, re.DOTALL)
                
                valid_results = []
                for json_obj in json_objects:
                    try:
                        data = json.loads(json_obj)
                        if "example" in data and "template" in data:
                            example = data["example"].strip()
                            new_template = data["template"].strip()
                            if example != original_log:
                                valid_results.append({"input": example, "output": new_template})
                    except:
                        continue
                
                if valid_results:
                    return valid_results
                
                # Fallback: split by lines and use original template
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]  
                return [{"input": log, "output": template} for log in potential_logs]  
                
        except Exception as e:  
            logger.error(f"Error parsing method 2 response: {e}")  
            # Fallback to line-by-line parsing  
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]  
            potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]
            return [{"input": log, "output": template} for log in potential_logs]

    def _parse_augmentation_response_method3(self, response_text, original_log, template):
        """
        Parse the augmented log response from API for method 3 (Semantic Rewriting)
        Method 3 generates completely new examples with their own templates
        """
        try:
            # Clean response text, extract JSON part
            response_text = response_text.strip()
            
            # Try to find JSON array part
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                
                # Extract examples and templates from JSON objects
                valid_results = []
                for item in json_data:
                    if isinstance(item, dict) and "example" in item and "template" in item:
                        example = item["example"].strip()
                        new_template = item["template"].strip()
                        # Skip if identical to original
                        if example != original_log:
                            valid_results.append({"input": example, "output": new_template})
                
                return valid_results
            else:
                # No JSON array found, try to extract JSON objects
                json_objects = re.findall(r'\{[^{}]*"example"[^{}]*"template"[^{}]*\}', response_text, re.DOTALL)
                
                valid_results = []
                for json_obj in json_objects:
                    try:
                        data = json.loads(json_obj)
                        if "example" in data and "template" in data:
                            example = data["example"].strip()
                            new_template = data["template"].strip()
                            if example != original_log:
                                valid_results.append({"input": example, "output": new_template})
                    except:
                        continue
                
                if valid_results:
                    return valid_results
                
                # Fallback: split by lines and use original template
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]  
                return [{"input": log, "output": template} for log in potential_logs]  
                
        except Exception as e:  
            logger.error(f"Error parsing method 3 response: {e}")  
            # Fallback to line-by-line parsing  
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]  
            potential_logs = [line for line in lines if not line.startswith(('```', '/*', '*/')) and len(line) > 5]
            return [{"input": log, "output": template} for log in potential_logs]

    def _get_additional_augmentations(self, original_log, template, error_summary, error_type, error_number, count, existing_results, method=1):  
        """  
        Get additional augmented logs when initial augmentations are insufficient  
        """  
        # Create more specific prompt based on the method
        if method == 1:
            specific_template = """  
Please generate {count} additional variable substitution variants for the following log. 
These must be different from the original log and already generated versions.
Only replace variable parts (<*>), keep constant parts exactly the same.

Original log: {original}  
Template: {template}  
Error Explanation: {error_summary}
Error Type: {error_type}

Already generated versions:  
{existing}  

Generate {count} new variants. Return as JSON array of strings only.
"""  
        elif method == 2:
            specific_template = """
Please generate {count} additional constant-inclusive rewriting variants.
These must be different from the original and already generated versions.
You can modify both constant and variable parts, and must provide new templates.

Original log: {original}  
Template: {template}  
Error Explanation: {error_summary}
Error Type: {error_type}

Already generated versions:  
{existing}  

Generate {count} new variants. Return as JSON array with "example" and "template" fields.
"""
        else:  # method 3
            specific_template = f"""
Please generate {{count}} additional semantic rewriting variants that trigger similar parsing errors.

{self.error_types_content}

Original log: {{original}}  
Template: {{template}}  
Error Explanation: {{error_summary}}
Error Type: {{error_type}}
Error Number: {{error_number}}

Already generated versions:  
{{existing}}  

Generate {{count}} new variants. Return as JSON array with "example", "template" and "error_type" fields.
"""
        
        # Format template
        specific_template = specific_template.format(
            count=count,
            original=original_log,
            template=template,
            error_summary=error_summary,
            error_type=error_type,
            error_number=error_number,
            existing="\n".join([f"- {result['input']}" for result in existing_results])
        )
        
        # Create messages  
        system_message = {"role": "system", "content": "You are a professional log augmentation tool."}  
        user_message = {"role": "user", "content": specific_template}  
        
        # Get additional augmentations  
        response_text = self._get_model_response([system_message, user_message])  
        
        # Parse response based on the method
        if method == 1:
            return self._parse_augmentation_response_method1(response_text, original_log, template)
        elif method == 2:
            return self._parse_augmentation_response_method2(response_text, original_log, template)
        else:  # method 3
            return self._parse_augmentation_response_method3(response_text, original_log, template)

# Usage example  
if __name__ == "__main__":  
    import argparse  
    
    # Command line arguments  
    parser = argparse.ArgumentParser(description='Augment log data using DeepSeek API based on research paper methods')  
    parser.add_argument('--api_key', type=str, help='DeepSeek API key (can also be set via DEEPSEEK_API_KEY environment variable)')  
    parser.add_argument('--model', type=str, default="deepseek-chat", help='DeepSeek model name')  
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file path with columns: Original_Log,Correct_Template,error_summary,Error_Type,Error_Number')  
    parser.add_argument('--output_file', type=str, help='Output JSON file path (defaults to input directory with augmented_ prefix)')  
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size (how many records to process before saving)')  
    parser.add_argument('--augmentation_count', type=int, default=7, help='Number of augmentations per log')  
    parser.add_argument('--error_types_file', type=str, required=True, help='Path to markdown file containing error types definitions (required)')
    
    args = parser.parse_args()  
    
    # Create augmentor  
    augmentor = DeepSeekLogAugmentor(  
        api_key=args.api_key,  
        model_name=args.model,  
        batch_size=args.batch_size,
        error_types_file=args.error_types_file
    )  
    augmentor.augmentation_count = args.augmentation_count  
    
    # Execute log augmentation  
    augmentor.augment_logs(args.input_file, args.output_file)