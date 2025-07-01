import pandas as pd
import numpy as np
import argparse
import logging
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeedSelector:
    def __init__(self):
        """
        Initialize seed selector
        """
        self.labeled_data = None
        self.summary_data = None
        self.selected_seeds = []

    def load_labeled_data(self, labeled_csv_path):
        """
        Load labeled data CSV file
        
        Args:
            labeled_csv_path: Path to CSV file with Original_Log, Correct_Template, Error_Type, Error_Number
        """
        if not os.path.exists(labeled_csv_path):
            raise FileNotFoundError(f"Labeled CSV file not found: {labeled_csv_path}")
        
        self.labeled_data = pd.read_csv(labeled_csv_path)
        logger.info(f"Loaded labeled data: {len(self.labeled_data)} records")
        
        # Check required columns
        required_columns = ['Original_Log', 'Correct_Template', 'Error_Type', 'Error_Number']
        missing_columns = [col for col in required_columns if col not in self.labeled_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in labeled data: {missing_columns}")
        
        # Show distribution of error types
        error_distribution = self.labeled_data['Error_Number'].value_counts().sort_index()
        logger.info("Error type distribution:")
        for error_num, count in error_distribution.items():
            logger.info(f"  Error Type {error_num}: {count} records")
        
        return self.labeled_data

    def load_summary_data(self, summary_csv_path):
        """
        Load summary data CSV file
        
        Args:
            summary_csv_path: Path to CSV file with original_log, correct_template, error_summary
        """
        if not os.path.exists(summary_csv_path):
            raise FileNotFoundError(f"Summary CSV file not found: {summary_csv_path}")
        
        self.summary_data = pd.read_csv(summary_csv_path)
        logger.info(f"Loaded summary data: {len(self.summary_data)} records")
        
        # Check required columns
        required_columns = ['original_log', 'correct_template', 'error_summary']
        missing_columns = [col for col in required_columns if col not in self.summary_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in summary data: {missing_columns}")
        
        # Create lookup dictionary for faster matching
        self.summary_lookup = {}
        for _, row in self.summary_data.iterrows():
            key = (str(row['original_log']).strip(), str(row['correct_template']).strip())
            self.summary_lookup[key] = str(row['error_summary'])
        
        logger.info(f"Created lookup dictionary with {len(self.summary_lookup)} entries")
        return self.summary_data

    def select_seeds_round_robin(self, num_seeds):
        """
        Select seeds using round-robin approach
        
        Args:
            num_seeds: Total number of seeds to select
            
        Returns:
            selected_seeds: List of selected seed records
        """
        if self.labeled_data is None:
            raise ValueError("Labeled data not loaded. Call load_labeled_data() first.")
        
        # Group data by error number
        grouped_data = {}
        error_numbers = sorted(self.labeled_data['Error_Number'].unique())
        
        for error_num in error_numbers:
            error_data = self.labeled_data[self.labeled_data['Error_Number'] == error_num].copy()
            # Shuffle the data for each error type to ensure randomness
            error_data = error_data.sample(frac=1, random_state=42).reset_index(drop=True)
            grouped_data[error_num] = error_data
        
        logger.info(f"Found {len(error_numbers)} error types: {error_numbers}")
        
        # Round-robin selection
        selected_seeds = []
        error_indices = {error_num: 0 for error_num in error_numbers}
        
        for i in range(num_seeds):
            # Determine which error type to select from
            current_error_num = error_numbers[i % len(error_numbers)]
            
            # Check if we have more data for this error type
            if error_indices[current_error_num] < len(grouped_data[current_error_num]):
                # Select the next record from this error type
                selected_record = grouped_data[current_error_num].iloc[error_indices[current_error_num]]
                selected_seeds.append(selected_record)
                error_indices[current_error_num] += 1
                
                logger.info(f"Selected seed {i+1}/{num_seeds}: Error Type {current_error_num}")
            else:
                logger.warning(f"No more data available for Error Type {current_error_num}")
        
        logger.info(f"Successfully selected {len(selected_seeds)} seeds")
        
        # Show selection distribution
        selection_distribution = defaultdict(int)
        for seed in selected_seeds:
            selection_distribution[seed['Error_Number']] += 1
        
        logger.info("Selection distribution:")
        for error_num in sorted(selection_distribution.keys()):
            logger.info(f"  Error Type {error_num}: {selection_distribution[error_num]} seeds")
        
        self.selected_seeds = selected_seeds
        return selected_seeds

    def match_with_summaries(self):
        """
        Match selected seeds with error summaries
        
        Returns:
            matched_seeds: List of seeds with error summaries
        """
        if not self.selected_seeds:
            raise ValueError("No seeds selected. Call select_seeds_round_robin() first.")
        
        if self.summary_data is None:
            raise ValueError("Summary data not loaded. Call load_summary_data() first.")
        
        matched_seeds = []
        unmatched_count = 0
        
        for i, seed in enumerate(self.selected_seeds):
            original_log = str(seed['Original_Log']).strip()
            correct_template = str(seed['Correct_Template']).strip()
            
            # Look up error summary
            lookup_key = (original_log, correct_template)
            error_summary = self.summary_lookup.get(lookup_key, "")
            
            if error_summary:
                matched_seed = {
                    'Original_Log': seed['Original_Log'],
                    'Correct_Template': seed['Correct_Template'],
                    'error_summary': error_summary,
                    'Error_Type': seed['Error_Type'],
                    'Error_Number': seed['Error_Number']
                }
                matched_seeds.append(matched_seed)
                logger.debug(f"Matched seed {i+1}: Found error summary")
            else:
                # Still include the seed but with empty error summary
                matched_seed = {
                    'Original_Log': seed['Original_Log'],
                    'Correct_Template': seed['Correct_Template'],
                    'error_summary': "",
                    'Error_Type': seed['Error_Type'],
                    'Error_Number': seed['Error_Number']
                }
                matched_seeds.append(matched_seed)
                unmatched_count += 1
                logger.warning(f"Seed {i+1}: No matching error summary found")
        
        logger.info(f"Matching completed. Matched: {len(matched_seeds) - unmatched_count}, Unmatched: {unmatched_count}")
        return matched_seeds

    def save_seeds(self, matched_seeds, output_path):
        """
        Save matched seeds to CSV file
        
        Args:
            matched_seeds: List of matched seed records
            output_path: Output CSV file path
        """
        if not matched_seeds:
            raise ValueError("No matched seeds to save")
        
        # Create DataFrame
        df = pd.DataFrame(matched_seeds)
        
        # Reorder columns
        column_order = ['Original_Log', 'Correct_Template', 'error_summary', 'Error_Type', 'Error_Number']
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Seeds saved to: {output_path}")
        
        # Show final statistics
        logger.info("Final seed statistics:")
        logger.info(f"  Total seeds: {len(df)}")
        logger.info(f"  Seeds with error summary: {len(df[df['error_summary'] != ''])}")
        logger.info(f"  Seeds without error summary: {len(df[df['error_summary'] == ''])}")
        
        # Show distribution by error type
        final_distribution = df['Error_Number'].value_counts().sort_index()
        logger.info("Final distribution by error type:")
        for error_num, count in final_distribution.items():
            logger.info(f"  Error Type {error_num}: {count} seeds")
        
        return df

    def run_seed_selection(self, labeled_csv_path, summary_csv_path, output_path, num_seeds):
        """
        Run complete seed selection process
        
        Args:
            labeled_csv_path: Path to labeled CSV file
            summary_csv_path: Path to summary CSV file
            output_path: Output CSV file path
            num_seeds: Number of seeds to select
        """
        logger.info(f"Starting seed selection process with {num_seeds} seeds")
        
        # Step 1: Load labeled data
        logger.info("Step 1: Loading labeled data")
        self.load_labeled_data(labeled_csv_path)
        
        # Step 2: Load summary data
        logger.info("Step 2: Loading summary data")
        self.load_summary_data(summary_csv_path)
        
        # Step 3: Select seeds using round-robin
        logger.info("Step 3: Selecting seeds using round-robin approach")
        selected_seeds = self.select_seeds_round_robin(num_seeds)
        
        # Step 4: Match with error summaries
        logger.info("Step 4: Matching seeds with error summaries")
        matched_seeds = self.match_with_summaries()
        
        # Step 5: Save results
        logger.info("Step 5: Saving results")
        result_df = self.save_seeds(matched_seeds, output_path)
        
        logger.info("Seed selection process completed successfully!")
        return result_df


def main():
    parser = argparse.ArgumentParser(description='Select seeds from labeled data using round-robin approach')
    parser.add_argument('--labeled_csv', type=str, required=True, 
                        help='Path to labeled CSV file (Original_Log, Correct_Template, Error_Type, Error_Number)')
    parser.add_argument('--summary_csv', type=str, required=True,
                        help='Path to summary CSV file (original_log, correct_template, error_summary)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to output CSV file')
    parser.add_argument('--num_seeds', type=int, required=True,
                        help='Number of seeds to select')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Create seed selector
    selector = SeedSelector()
    
    # Run seed selection process
    result_df = selector.run_seed_selection(
        labeled_csv_path=args.labeled_csv,
        summary_csv_path=args.summary_csv,
        output_path=args.output_csv,
        num_seeds=args.num_seeds
    )
    
    logger.info("=" * 50)
    logger.info("Seed Selection Summary:")
    logger.info(f"Input labeled file: {args.labeled_csv}")
    logger.info(f"Input summary file: {args.summary_csv}")
    logger.info(f"Output file: {args.output_csv}")
    logger.info(f"Number of seeds requested: {args.num_seeds}")
    logger.info(f"Number of seeds generated: {len(result_df)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()