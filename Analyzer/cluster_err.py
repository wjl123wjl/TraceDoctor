import os
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorTypeMerger:
    def __init__(self, openai_api_key=None, openai_base_url=None):
        """
        Initialize error type merger
        
        Args:
            openai_api_key: OpenAI API key
            openai_base_url: OpenAI API base URL (optional)
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        if openai_base_url:
            self.client = OpenAI(api_key=self.openai_api_key, base_url=openai_base_url)
        else:
            self.client = OpenAI(api_key=self.openai_api_key)
        
        self.error_types = {}
        self.embeddings = {}
        self.similarity_threshold = 0.9

    def parse_md_file(self, md_file_path):
        """
        Parse MD file to extract error types
        
        Args:
            md_file_path: Path to MD file
            
        Returns:
            error_types: Dictionary with format {letter: {title: str, description: str, examples: list}}
        """
        if not os.path.exists(md_file_path):
            raise FileNotFoundError(f"MD file not found: {md_file_path}")
        
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        error_types = {}
        
        # Use regex to match each error type
        pattern = r'#### ([A-Z])\. ([^\n]+)\n- ([^#]+?)(?=####|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for letter, title, description_block in matches:
            # Clean description text
            description_block = description_block.strip()
            
            # Separate description and examples
            lines = description_block.split('\n')
            description_lines = []
            examples = []
            
            in_examples = False
            for line in lines:
                line = line.strip()
                if line.startswith('- Example errors:') or line.startswith('- Examples:'):
                    in_examples = True
                    continue
                elif line.startswith('- "') and in_examples:
                    # Extract examples from quotes
                    example = line[3:-1] if line.endswith('"') else line[3:]
                    examples.append(example)
                elif not in_examples and line.startswith('- '):
                    # Description part
                    description_lines.append(line[2:])
                elif not line.startswith('-') and not in_examples:
                    description_lines.append(line)
            
            error_types[letter] = {
                'title': title.strip(),
                'description': ' '.join(description_lines).strip(),
                'examples': examples
            }
        
        logger.info(f"Successfully parsed {len(error_types)} error types: {list(error_types.keys())}")
        self.error_types = error_types
        return error_types

    def get_embeddings(self):
        """
        Get embedding vectors for error types
        """
        logger.info("Starting to get embedding vectors for error types...")
        
        for letter, error_info in self.error_types.items():
            # Combine title, description, and examples as text
            text_parts = [
                error_info['title'],
                error_info['description']
            ]
            
            # Add some examples (avoid text being too long)
            if error_info['examples']:
                text_parts.extend(error_info['examples'][:2])  # Only take first two examples
            
            combined_text = ' '.join(text_parts)
            
            # Call OpenAI embedding API
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Use smaller model
                input=combined_text
            )
            
            self.embeddings[letter] = response.data[0].embedding
            logger.info(f"Successfully got embedding vector for error type {letter}")
        
        logger.info("All error type embedding vectors obtained successfully")

    def calculate_similarity_matrix(self):
        """
        Calculate cosine similarity matrix between error types
        
        Returns:
            similarity_matrix: Similarity matrix
            letters: Corresponding letter list
        """
        letters = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[letter] for letter in letters])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        logger.info("Similarity matrix calculation completed")
        logger.info(f"Matrix shape: {similarity_matrix.shape}")
        
        return similarity_matrix, letters

    def find_similar_pairs(self, similarity_matrix, letters):
        """
        Find category pairs with similarity above threshold
        
        Args:
            similarity_matrix: Similarity matrix
            letters: Letter list
            
        Returns:
            similar_pairs: List of similar category pairs
        """
        similar_pairs = []
        
        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                similarity = similarity_matrix[i][j]
                if similarity > self.similarity_threshold:
                    similar_pairs.append((letters[i], letters[j], similarity))
                    logger.info(f"Found similar category pair: {letters[i]} - {letters[j]}, similarity: {similarity:.4f}")
        
        return similar_pairs

    def analyze_csv_statistics(self, csv_file_path):
        """
        Analyze statistics of error types in CSV file
        
        Args:
            csv_file_path: CSV file path
            
        Returns:
            type_counts: Count statistics for each type
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded CSV file: {csv_file_path}, total {len(df)} records")
        
        # Count each error type
        type_counts = df['Error_Type'].value_counts().to_dict()
        
        logger.info("Error type statistics:")
        for error_type, count in type_counts.items():
            logger.info(f"  {error_type}: {count}")
        
        return type_counts, df

    def create_merge_plan(self, similar_pairs, type_counts):
        """
        Create merge plan: merge categories with fewer instances into those with more
        
        Args:
            similar_pairs: Similar category pairs
            type_counts: Count statistics for each type
            
        Returns:
            merge_plan: Merge plan dictionary {from_type: to_type}
        """
        merge_plan = {}
        
        # Create mapping from letter to error type name
        letter_to_type = {}
        for letter, error_info in self.error_types.items():
            letter_to_type[letter] = error_info['title']
        
        for letter1, letter2, similarity in similar_pairs:
            type1 = letter_to_type[letter1]
            type2 = letter_to_type[letter2]
            
            count1 = type_counts.get(type1, 0)
            count2 = type_counts.get(type2, 0)
            
            # Merge the one with fewer instances into the one with more
            if count1 > count2:
                merge_plan[type2] = type1
                logger.info(f"Merge plan: {type2} ({count2}) -> {type1} ({count1})")
            else:
                merge_plan[type1] = type2
                logger.info(f"Merge plan: {type1} ({count1}) -> {type2} ({count2})")
        
        return merge_plan

    def apply_merge_to_csv(self, df, merge_plan, output_csv_path):
        """
        Apply merge plan to CSV file
        
        Args:
            df: Original DataFrame
            merge_plan: Merge plan
            output_csv_path: Output CSV file path
        """
        # Create copy to avoid modifying original data
        new_df = df.copy()
        
        # Apply merge plan
        for from_type, to_type in merge_plan.items():
            mask = new_df['Error_Type'] == from_type
            new_df.loc[mask, 'Error_Type'] = to_type
            logger.info(f"Merged type: {from_type} -> {to_type}, affected {mask.sum()} records")
        
        # Re-count type distribution after merge
        new_type_counts = new_df['Error_Type'].value_counts().to_dict()
        logger.info("Error type statistics after merge:")
        for error_type, count in new_type_counts.items():
            logger.info(f"  {error_type}: {count}")
        
        # Save new CSV file
        new_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"New CSV file saved: {output_csv_path}")
        
        return new_df, new_type_counts

    def generate_new_md_file(self, merge_plan, output_md_path):
        """
        Generate new MD file, excluding merged types
        
        Args:
            merge_plan: Merge plan
            output_md_path: Output MD file path
        """
        # Get merged types (need to exclude)
        merged_types = set(merge_plan.keys())
        
        # Build new MD content
        md_content = """# Log Parsing Error Type Analysis (Merged)

Analysis results after similarity-based merging
Similarity threshold: 0.9

## Data Overview

- Error types merged based on semantic similarity
- Categories with similarity > 0.9 have been consolidated

---

## Error Type Classification

Here are the consolidated error categories:

"""
        
        # Reassign letter identifiers
        letter_counter = 0
        for letter, error_info in self.error_types.items():
            error_type = error_info['title']
            
            # Skip merged types
            if error_type in merged_types:
                continue
            
            current_letter = chr(ord('A') + letter_counter)
            letter_counter += 1
            
            md_content += f"#### {current_letter}. {error_info['title']}\n"
            md_content += f"- {error_info['description']}\n"
            
            if error_info['examples']:
                md_content += "- Example errors:\n"
                for example in error_info['examples']:
                    md_content += f'  - "{example}"\n'
            
            md_content += "\n"
        
        # Save new MD file
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"New MD file saved: {output_md_path}")

    def run_merge_process(self, md_file_path, csv_file_path, output_dir):
        """
        Run complete merge process
        
        Args:
            md_file_path: Input MD file path
            csv_file_path: Input CSV file path
            output_dir: Output directory
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Parse MD file
        logger.info("Step 1: Parse MD file")
        self.parse_md_file(md_file_path)
        
        # 2. Get embedding vectors
        logger.info("Step 2: Get embedding vectors")
        self.get_embeddings()
        
        # 3. Calculate similarity matrix
        logger.info("Step 3: Calculate similarity matrix")
        similarity_matrix, letters = self.calculate_similarity_matrix()
        
        # 4. Find similar category pairs
        logger.info("Step 4: Find similar category pairs")
        similar_pairs = self.find_similar_pairs(similarity_matrix, letters)
        
        if not similar_pairs:
            logger.info("No category pairs found with similarity above threshold, no merge needed")
            return
        
        # 5. Analyze CSV statistics
        logger.info("Step 5: Analyze CSV statistics")
        type_counts, df = self.analyze_csv_statistics(csv_file_path)
        
        # 6. Create merge plan
        logger.info("Step 6: Create merge plan")
        merge_plan = self.create_merge_plan(similar_pairs, type_counts)
        
        if not merge_plan:
            logger.info("No merge plan generated, no merge needed")
            return
        
        # 7. Apply merge to CSV
        logger.info("Step 7: Apply merge to CSV file")
        output_csv_path = os.path.join(output_dir, "merged_labels.csv")
        new_df, new_type_counts = self.apply_merge_to_csv(df, merge_plan, output_csv_path)
        
        # 8. Generate new MD file
        logger.info("Step 8: Generate new MD file")
        output_md_path = os.path.join(output_dir, "merged_error_types.md")
        self.generate_new_md_file(merge_plan, output_md_path)
        
        logger.info("Merge process completed!")
        return {
            'merge_plan': merge_plan,
            'output_csv': output_csv_path,
            'output_md': output_md_path,
            'new_type_counts': new_type_counts
        }


def main():
    parser = argparse.ArgumentParser(description='Merge similar error types based on semantic similarity')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key')
    parser.add_argument('--openai_base_url', type=str, help='OpenAI API base URL (optional)')
    parser.add_argument('--md_file', type=str, required=True, help='Input MD file path')
    parser.add_argument('--csv_file', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--similarity_threshold', type=float, default=0.9, help='Similarity threshold for merging')
    
    args = parser.parse_args()
    
    # Create error type merger
    merger = ErrorTypeMerger(
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url
    )
    
    # Set similarity threshold
    merger.similarity_threshold = args.similarity_threshold
    
    # Run merge process
    results = merger.run_merge_process(
        md_file_path=args.md_file,
        csv_file_path=args.csv_file,
        output_dir=args.output_dir
    )
    
    if results:
        logger.info("=" * 50)
        logger.info("Merge Results Summary:")
        logger.info(f"Output CSV file: {results['output_csv']}")
        logger.info(f"Output MD file: {results['output_md']}")
        logger.info(f"Merge plan: {results['merge_plan']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()