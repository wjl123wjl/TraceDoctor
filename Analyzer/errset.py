import os  
import csv  
import pandas as pd  
from tqdm import tqdm  
import time  
import logging  
import re  
import requests  
from tenacity import retry, stop_after_attempt, wait_random_exponential  
import argparse

# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

class DeepSeekLogErrorAnalyzer:  
    def __init__(self, api_key, model_name="deepseek-chat", batch_size=5):  
        self.batch_size = batch_size  
        self.model_name = model_name  
        self.api_key = api_key
        
        if not self.api_key:  
            raise ValueError("DeepSeek API key is required. Please provide it via --api_key parameter or DEEPSEEK_API_KEY environment variable")  
        
        # API endpoint  
        self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"  
        
        logger.info(f"Initialized DeepSeek API (model: {model_name})")  
        
        # Error analysis prompt template
        self.prompt_template = """  
You are an expert in log parsing analysis. Your task is to analyze how a student model processes log templates and identify errors in their reasoning.  

For each log, I will provide:  
1. Original log: The raw log message  
2. Student's thinking process: How they reasoned about extracting the template  
3. Correct template: The expected template with variables marked as <*>  

Analyze the student's reasoning and identify any errors they made. DO NOT classify the error type.  
Instead, provide a detailed description of what went wrong in the reasoning process (2-3 sentences).  

IMPORTANT: Your output must EXACTLY follow this format with NOTHING ELSE:  
error_summary: A detailed explanation of what went wrong (or what was done correctly) in 2-3 sentences.  
"""  

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))  
    def _call_deepseek_api(self, messages, temperature=0.1):  
        """  
        Call DeepSeek API and handle possible errors  
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
            "top_p": 0.9,  
            "stream": False  
        }  
        
        try:  
            response = requests.post(self.api_endpoint, json=payload, headers=headers)  
            response.raise_for_status()  
            return response.json()  
        except requests.exceptions.RequestException as e:  
            logger.error(f"API request error: {e}")  
            if response.status_code == 429:  
                logger.warning("Rate limit reached, retrying...")  
                time.sleep(10)  
            elif response.status_code == 500:  
                logger.error("Server error, retrying...")  
            raise  

    def process_csv(self, input_file, output_file):  
        # Check input file  
        if not os.path.exists(input_file):  
            logger.error(f"Input file not found: {input_file}")  
            return  
        
        # Read input CSV file  
        try:  
            df = pd.read_csv(input_file)  
            logger.info(f"Loaded {len(df)} records from {input_file}")  
        except Exception as e:  
            logger.error(f"Error reading input file: {e}")  
            return  
        
        # Create output CSV file and write headers  
        csv_columns = ['original_log', 'correct_template', 'error_summary']  
        
        # Check if file already exists  
        file_exists = os.path.isfile(output_file)  
        
        # If it doesn't exist, create file and write headers  
        if not file_exists:  
            try:  
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:  
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns, quoting=csv.QUOTE_ALL)  
                    writer.writeheader()  
                logger.info(f"Created output file: {output_file}")  
            except Exception as e:  
                logger.error(f"Error creating output file: {e}")  
                return  
        else:  
            logger.info(f"Output file {output_file} already exists, will append results")  
        
        # Progress tracking  
        processed_count = 0  
        total_count = len(df)  
        start_time = time.time()  
        current_batch_results = []  
        
        # Process records in batches  
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):  
            try:  
                # Ensure column names match  
                required_columns = ['Content', 'Think-Process', 'EventTemplate']  
                for col in required_columns:  
                    if col not in row.index:  
                        logger.warning(f"Missing column: {col}. Available columns: {row.index.tolist()}")  
                        raise KeyError(f"Missing required column: {col}")  
                
                content = row['Content']  
                think_process = row['Think-Process']  
                event_template = row['EventTemplate']  
                
                # Skip empty fields  
                if pd.isna(content) or pd.isna(think_process) or pd.isna(event_template):  
                    logger.warning("Skipping row with empty fields")  
                    continue  
                
                # Create prompt  
                messages = self._create_messages(content, think_process, event_template)  
                
                # Get model response  
                logger.debug(f"Processing log: {content[:50]}...")  
                start_time_record = time.time()  
                response = self._get_model_response(messages)  
                inference_time = time.time() - start_time_record  
                logger.debug(f"Inference time: {inference_time:.2f} seconds")  
                
                # Parse response  
                analysis = self._parse_response(response)  
                
                # Log raw response and parsed summary for debugging  
                logger.debug(f"Raw response: {response}")  
                logger.debug(f"Parsed summary: {analysis.get('error_summary', 'N/A')}")  
                
                # Create result entry  
                result = {  
                    'original_log': content,  
                    'correct_template': event_template,  
                    'error_summary': analysis.get('error_summary', '')  
                }  
                
                # Output brief real-time information  
                logger.info(f"Log: '{content[:30]}...' → Summary: {analysis.get('error_summary', 'N/A')[:50]}...")  
                
                # Add to current batch  
                current_batch_results.append(result)  
                processed_count += 1  
                
                # Write to CSV every batch_size records  
                if len(current_batch_results) >= self.batch_size or i == len(df) - 1:  
                    # Write current batch to CSV  
                    try:  
                        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:  
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, quoting=csv.QUOTE_ALL)  
                            for result_item in current_batch_results:  
                                writer.writerow(result_item)  
                        
                        # Update progress information  
                        elapsed_time = time.time() - start_time  
                        records_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0  
                        estimated_remaining = (total_count - processed_count) / records_per_second if records_per_second > 0 else 0  
                        
                        logger.info(f"Written {len(current_batch_results)} records. Total progress: {processed_count}/{total_count} "  
                                  f"({processed_count/total_count*100:.1f}%) - "  
                                  f"Speed: {records_per_second:.2f} records/sec - "  
                                  f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")  
                        
                        # Clear current batch  
                        current_batch_results = []  
                        
                    except Exception as e:  
                        logger.error(f"Error writing results to CSV file: {e}")  
                
            except Exception as e:  
                logger.error(f"Error processing record: {e}")  
                import traceback  
                logger.error(traceback.format_exc())  
                continue  
        
        total_time = time.time() - start_time  
        logger.info(f"Analysis completed. Total processed records: {processed_count}/{total_count} "  
                  f"Time taken: {total_time/60:.1f} minutes "  
                  f"(Average: {processed_count/total_time:.2f} records/sec)")  
    
    def _create_messages(self, content, think_process, event_template):  
        """  
        Create message format required by DeepSeek API  
        """  
        # System message uses prompt template  
        system_message = {"role": "system", "content": self.prompt_template}  
        
        # User message contains specific log analysis request  
        user_content = f"Original log: {content}\nStudent's thinking process: {think_process}\nCorrect template: {event_template}\n\nAnalysis:"  
        user_message = {"role": "user", "content": user_content}  
        
        return [system_message, user_message]  
    
    def _get_model_response(self, messages):  
        """  
        Get model response through DeepSeek API  
        """  
        try:  
            # Call API  
            response_json = self._call_deepseek_api(messages, temperature=0.1)  
            
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
    
    def _parse_response(self, response):  
        try:  
            # Storage for parsing results  
            analysis = {}  
            
            # Initial response cleanup  
            # 1. Remove all thinking markers  
            response = response.replace("</think>", " ")  
            
            # 2. Remove markdown code block markers  
            response = re.sub(r"```.*?```", " ", response, flags=re.DOTALL)  
            
            # 3. Remove all Step lines  
            response = re.sub(r"## Step \d+:.*?\n", " ", response)  
            
            # 4. Remove boxed markers  
            response = re.sub(r"\$\\boxed\{(.*?)\}\$", r"\1", response, flags=re.DOTALL)  
            response = re.sub(r"\$?\\boxed\{(.*?)\}\$?", r"\1", response, flags=re.DOTALL)  
            
            # 5. Find the last error_summary definition  
            if "error_summary:" in response.lower():  
                # Find all matching error_summary instances  
                summaries = re.findall(r"error_summary:(.*?)(?=error_summary:|$)",   
                                      response.lower() + "error_summary:", re.DOTALL)  
                
                if summaries:  
                    # Get the last summary (final answer)  
                    last_summary = summaries[-1].strip()  
                    
                    # Clean quotes and extra spaces  
                    last_summary = re.sub(r'^["\'\s]+|["\'\s]+$', '', last_summary)  
                    
                    # Remove excess whitespace  
                    last_summary = ' '.join(last_summary.split())  
                    
                    # Store cleaned summary  
                    analysis['error_summary'] = last_summary  
            
            # If no valid summary found, perform fallback parsing  
            if 'error_summary' not in analysis or not analysis['error_summary']:  
                # Try to find the last continuous text paragraph as summary  
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', response) if p.strip()]  
                if paragraphs:  
                    analysis['error_summary'] = paragraphs[-1]  
            
            # Ensure summary doesn't exceed reasonable length  
            if 'error_summary' in analysis and len(analysis['error_summary']) > 1000:  
                analysis['error_summary'] = analysis['error_summary'][:997] + "..."  
            
            return analysis  
            
        except Exception as e:  
            logger.error(f"Error parsing response: {e}")  
            import traceback  
            logger.error(traceback.format_exc())  
            return {}  

def main():
    parser = argparse.ArgumentParser(description='Analyze log parsing errors using DeepSeek API')  
    parser.add_argument('--api_key', type=str, 
                        help='DeepSeek API key (can also be set via DEEPSEEK_API_KEY environment variable)')  
    parser.add_argument('--model', type=str, default="deepseek-chat", 
                        help='DeepSeek model name')  
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file path')  
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output CSV file path')  
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Batch processing size')  
    
    args = parser.parse_args()  
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DeepSeek API key is required. Please provide it via --api_key parameter or set DEEPSEEK_API_KEY environment variable")
        return
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Input file: {args.input_file}")
    logger.info(f"  Output file: {args.output_file}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    # Create analyzer  
    analyzer = DeepSeekLogErrorAnalyzer(  
        api_key=api_key,  
        model_name=args.model,  
        batch_size=args.batch_size  
    )  
    
    # Process CSV file  
    analyzer.process_csv(args.input_file, args.output_file)

if __name__ == "__main__":  
    main()
