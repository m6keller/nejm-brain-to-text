import os
import pandas as pd
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

# Assuming these are available in your directory as per your original script
from utils import DATA_BASE_PATH
from evaluate_model_helpers import load_h5py_file

def main():
    parser = argparse.ArgumentParser(description='Extract Ground Truth CSV from CopyTask Dataset.')
    parser.add_argument('--args-path', type=str, default="rnn_args.yaml",
                        help='Path to model args (to load args.yaml for session lists).')
    parser.add_argument('--data-dir', type=str, default=str(DATA_BASE_PATH / 't15_copyTask_neuralData/hdf5_data_final/'),
                        help='Path to the dataset directory.')
    parser.add_argument('--csv-path', type=str, default='../data/t15_copyTaskData_description.csv',
                        help='Path to the CSV file with metadata about the dataset.')
    parser.add_argument('--output-name', type=str, default='validation_ground_truth.csv',
                        help='Name of the output CSV file.')

    args = parser.parse_args()

    # 1. Load Configuration (to get the list of sessions used in training)
    print(f"Loading config from {args.args_path}...")
    model_args = OmegaConf.load(args.args_path)

    # 2. Load Metadata CSV
    b2txt_csv_df = pd.read_csv(args.csv_path)

    # 3. Load Data from HDF5
    print("Loading validation data...")
    test_data = {}
    total_trials = 0
    
    # We iterate through the sessions defined in the model args to ensure order consistency
    for session in model_args['dataset']['sessions']:
        files = [f for f in os.listdir(os.path.join(args.data_dir, session)) if f.endswith('.hdf5')]
        if 'data_val.hdf5' in files:
            eval_file = os.path.join(args.data_dir, session, 'data_val.hdf5')

            # Use the helper function from your original script
            data = load_h5py_file(eval_file, b2txt_csv_df)
            test_data[session] = data
            
            total_trials += len(data["neural_features"])
            print(f'Loaded {len(data["neural_features"])} trials for session {session}.')

    # 4. Extract Sentences
    print("Extracting ground truth sentences...")
    
    extracted_ids = []
    extracted_sentences = []
    
    # Global ID counter to match the sequential ID in your example
    global_id = 0

    for session in model_args['dataset']['sessions']:
        if session in test_data:
            data = test_data[session]
            num_trials = len(data['neural_features'])
            
            for trial in range(num_trials):
                # Extract the ground truth sentence label
                # 'sentence_label' is usually populated by load_h5py_file using the metadata CSV
                sentence = data['sentence_label'][trial]
                
                # Handle cases where sentence might be None or need cleaning
                if sentence is None:
                    sentence = ""
                else:
                    sentence = str(sentence).strip()

                extracted_ids.append(global_id)
                extracted_sentences.append(sentence)
                
                global_id += 1

    df_out = pd.DataFrame({
        'id': extracted_ids, 
        'text': extracted_sentences
    })
    
    output_path = args.output_name
    df_out.to_csv(output_path, index=False)
    
    print(f"Successfully saved {len(df_out)} items to {output_path}")

if __name__ == "__main__":
    main()