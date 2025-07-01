import os
import re
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogErrorAnalyzer:
    def __init__(self, api_key, model_name="deepseek-reasoner"):
        self.model_name = model_name
        self.api_key = api_key
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Please provide it via --api_key parameter or DEEPSEEK_API_KEY environment variable")
        
        # Choose API endpoint based on model type
        if "reasoner" in model_name.lower():
            self.api_endpoint = "https://api.deepseek.com/beta/chat/completions"
            logger.info(f"Using reasoning model endpoint: {self.api_endpoint}")
        else:
            self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
            logger.info(f"Using standard endpoint: {self.api_endpoint}")
        
        logger.info(f"Initialized DeepSeek API (model: {model_name})")
        
        # Modified error analysis template
        self.error_analysis_template = """
You are an expert in log parsing analysis. I will provide you with error summaries from failed log parsing attempts.

Your task is to:
1. Analyze the error patterns and naturally group them into distinct error categories
2. For each category, provide a clear name and description
3. Include 2-3 SPECIFIC ERROR TEXT EXAMPLES from the actual data provided
4. Do NOT include frequency analysis, recommendations, or priority rankings
5. Focus ONLY on categorizing and describing the error types

Below are the error summaries from log parsing attempts:

{error_summaries}

Please provide your analysis in the following format for each error category you identify:

#### [Letter]. [Error Category Name]
- [Brief description of what this error type involves]
- Example errors: [Quote 2-3 actual error text examples from the data above]

IMPORTANT GUIDELINES:
- Group errors naturally
- Focus on the core parsing logic errors, not infrastructure issues
- Provide clear, descriptive category names
- Include ACTUAL ERROR TEXT as examples, not numbers or IDs
- Quote the exact error text from the summaries I provided
- Do NOT include frequency counts, impact analysis, or recommendations
- Only output the error categories and their descriptions with actual text examples

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
        
        # Adjust parameters for reasoning model
        if "reasoner" in self.model_name.lower():
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000,
                "top_p": 0.9,
                "stream": False
            }
        else:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 3000,
                "top_p": 0.9,
                "stream": False
            }
        
        try:
            logger.info(f"Sending API request to: {self.api_endpoint}")
            logger.info(f"Using model: {self.model_name}")
            
            response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=120)
            
            logger.info(f"API response status code: {response.status_code}")
            
            # If 404 error, try other endpoint
            if response.status_code == 404 and "reasoner" in self.model_name.lower():
                logger.warning("Beta endpoint failed, trying v1 endpoint...")
                self.api_endpoint = "https://api.deepseek.com/v1/chat/completions"
                response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=120)
                logger.info(f"v1 endpoint response status code: {response.status_code}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            if hasattr(response, 'status_code'):
                logger.error(f"HTTP status code: {response.status_code}")
                if response.status_code == 404:
                    logger.error("404 error: API endpoint does not exist")
                elif response.status_code == 401:
                    logger.error("401 error: Invalid API key")
                elif response.status_code == 403:
                    logger.error("403 error: No permission to access this model")
                elif response.status_code == 429:
                    logger.warning("429 error: Rate limit reached")
                elif response.status_code == 500:
                    logger.error("500 error: Internal server error")
                
                try:
                    error_detail = response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    logger.error(f"Response content: {response.text[:500]}...")
            raise

    def _get_model_response(self, messages):
        """
        Get model response through DeepSeek API
        """
        try:
            response_json = self._call_deepseek_api(messages, temperature=0.1)
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                choice = response_json['choices'][0]
                
                # Check for special response format of reasoning model
                if 'message' in choice:
                    response_text = choice['message']['content']
                elif 'reasoning_content' in choice:
                    response_text = choice['reasoning_content']
                else:
                    logger.warning(f"Unexpected response format: {choice}")
                    response_text = str(choice)
                
                return response_text
            else:
                logger.error(f"Abnormal API response format: {response_json}")
                return ""
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return ""

    def analyze_csv_errors(self, csv_file, output_file, sample_size=None, error_column='error_summary'):
        """
        Analyze error summaries in CSV file and generate error type summary
        """
        try:
            # Read CSV file
            if not os.path.exists(csv_file):
                logger.error(f"CSV file does not exist: {csv_file}")
                return False
            
            df = pd.read_csv(csv_file)
            logger.info(f"Successfully read CSV file with {len(df)} records")
            
            # Check if error summary column exists
            if error_column not in df.columns:
                logger.error(f"Column '{error_column}' does not exist in CSV file. Available columns: {df.columns.tolist()}")
                return False
            
            # Extract error summaries
            error_summaries = df[error_column].dropna().tolist()
            logger.info(f"Extracted {len(error_summaries)} non-empty error summaries")
            
            if len(error_summaries) == 0:
                logger.error("No valid error summaries found")
                return False
            
            # Filter and clean error summaries
            cleaned_summaries = []
            for summary in error_summaries:
                if isinstance(summary, str) and len(summary.strip()) > 10:
                    cleaned_summary = summary.strip()
                    cleaned_summaries.append(cleaned_summary)
            
            logger.info(f"After cleaning, {len(cleaned_summaries)} valid error summaries remain")
            
            # Sample processing - reasoning model may need fewer samples
            max_samples = 200 if "reasoner" in self.model_name.lower() else 300
            
            if sample_size is not None and sample_size < len(cleaned_summaries):
                import random
                random.seed(42)
                sampled_summaries = random.sample(cleaned_summaries, sample_size)
                logger.info(f"Randomly sampled {sample_size} error summaries for analysis")
            elif len(cleaned_summaries) > max_samples:
                import random
                random.seed(42)
                sampled_summaries = random.sample(cleaned_summaries, max_samples)
                logger.info(f"Automatically limited to {max_samples} for analysis (to avoid oversized requests)")
            else:
                sampled_summaries = cleaned_summaries
                logger.info(f"Analyzing all {len(sampled_summaries)} error summaries")
            
            # Build analysis input - provide error text directly without numbering
            error_summaries_text = ""
            for summary in sampled_summaries:
                error_summaries_text += f"- {summary}\n\n"
            
            # Create API messages
            system_message = {
                "role": "system", 
                "content": "You are an expert in log parsing analysis who specializes in categorizing parsing errors into clear, distinct types based on actual error content."
            }
            user_content = self.error_analysis_template.format(
                error_summaries=error_summaries_text
            )
            user_message = {"role": "user", "content": user_content}
            messages = [system_message, user_message]
            
            logger.info(f"Generating error type analysis using {self.model_name}...")
            
            # Get model response
            error_analysis = self._get_model_response(messages)
            
            if not error_analysis:
                logger.error("Failed to get valid analysis results")
                return False
            
            # Create output directory
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Write analysis results
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Log Parsing Error Type Analysis\n\n")
                f.write(f"Analysis results based on {len(sampled_summaries)} error summaries\n")
                f.write(f"Model used: {self.model_name}\n\n")
                f.write("## Data Overview\n\n")
                f.write(f"- Data source: {os.path.basename(csv_file)}\n")
                f.write(f"- Total records: {len(df)}\n")
                f.write(f"- Valid error summaries: {len(cleaned_summaries)}\n")
                f.write(f"- Analysis sample size: {len(sampled_summaries)}\n\n")
                f.write("---\n\n")
                f.write("## Error Type Classification\n\n")
                f.write(error_analysis)
            
            logger.info(f"Error type analysis generated: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    parser = argparse.ArgumentParser(description='Analyze log parsing errors in CSV file and generate error type summary')
    parser.add_argument('--api_key', type=str, 
                        help='DeepSeek API key (can also be set via DEEPSEEK_API_KEY environment variable)')
    parser.add_argument('--model', type=str, default="deepseek-reasoner", 
                        help='DeepSeek model name (deepseek-reasoner or deepseek-chat)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='CSV file path containing error summaries')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output error type analysis file path')
    parser.add_argument('--sample_size', type=int, default=500, 
                        help='Sample size for analysis')
    parser.add_argument('--error_column', type=str, default='error_summary', 
                        help='Column name containing error summaries in CSV file')
    
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DeepSeek API key is required. Please provide it via --api_key parameter or set DEEPSEEK_API_KEY environment variable")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input CSV file does not exist: {args.input_file}")
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
    logger.info(f"  Sample size: {args.sample_size}")
    logger.info(f"  Error column: {args.error_column}")
    
    # Create analyzer
    analyzer = LogErrorAnalyzer(
        api_key=api_key,
        model_name=args.model
    )
    
    # Perform error analysis
    success = analyzer.analyze_csv_errors(
        csv_file=args.input_file,
        output_file=args.output_file,
        sample_size=args.sample_size,
        error_column=args.error_column
    )
    
    if success:
        logger.info("Error type analysis completed successfully!")
    else:
        logger.error("Error type analysis failed!")

if __name__ == "__main__":
    main()
