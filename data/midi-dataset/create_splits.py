import csv
import random
import numpy as np

def generate_data_splits_no_repeats(input_file, 
                                    training_size=0.7, 
                                    val_size=0.15,  # Same size for both val1 and val2
                                    test_size=0.05,
                                    val2_instruments=['001', '002', '003'],
                                    path_prefix='/mnt/vdb/random_audios_patch_16k/'):
    
    # Generate all possible melody and instrument combinations
    all_melody_ids = [str(x) for x in range(101, 1000)]
    all_instrument_ids = [str(x).zfill(3) for x in range(1, 105)]
    all_pairs = [(melody, inst) for melody in all_melody_ids for inst in all_instrument_ids]
    all_pairs_set = set(all_pairs)
    
    # Separate pairs for Val2 and the rest
    val2_style_pairs = [(melody, inst) for melody in all_melody_ids for inst in val2_instruments]
    non_val2_pairs = [(melody, inst) for melody in all_melody_ids for inst in all_instrument_ids if inst not in val2_instruments]
    
    # Shuffle and split non Val2 pairs
    np.random.shuffle(non_val2_pairs)
    n = len(non_val2_pairs)
    train_end = int(n * training_size)
    val_end = train_end + int(n * val_size)
    
    train_pairs = non_val2_pairs[:train_end]
    val_pairs = non_val2_pairs[train_end:val_end]
    test_pairs = non_val2_pairs[val_end:]
    
    # Split Val pairs into Val1 and Val2
    val1_pairs = val_pairs[:len(val_pairs)//2]
    val2_pairs = val_pairs[len(val_pairs)//2:]

    # Helper function to generate CSV files
    def generate_csv_rows_no_repeats(pairs, style_pairs, all_pairs_set, output_file):
        used_files = set()
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for content in pairs:
                if content in used_files:
                    continue
                style = random.choice(style_pairs)
                ground_truth = (content[0], style[1])
                if ground_truth in all_pairs_set:
                    writer.writerow([f"{path_prefix}/data_{content[0]}_{content[1]}.wav",
                                     f"{path_prefix}/data_{style[0]}_{style[1]}.wav",
                                     f"{path_prefix}/data_{ground_truth[0]}_{ground_truth[1]}.wav"])
                    used_files.add(content)
                    used_files.add(style)
                    used_files.add(ground_truth)
    
    # Generate CSV files for each set
    generate_csv_rows_no_repeats(train_pairs, non_val2_pairs, all_pairs_set, 'training_set.csv')
    generate_csv_rows_no_repeats(val1_pairs, non_val2_pairs, all_pairs_set, 'validation_set_1.csv')
    generate_csv_rows_no_repeats(val2_pairs, val2_style_pairs, all_pairs_set, 'validation_set_2.csv')
    generate_csv_rows_no_repeats(test_pairs, non_val2_pairs, all_pairs_set, 'test_set.csv')
    
    print("CSV files generated: training_set.csv, validation_set_1.csv, validation_set_2.csv, test_set.csv")

# Entry point
if __name__ == "__main__":
    # Confirmation check
    proceed = input("Do you want to proceed with creating dataset splits? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Operation cancelled.")
    else:
        generate_data_splits_no_repeats('file_list.txt', val2_instruments=[str(x).zfill(3) for x in range(41, 51)], val_size=0.02, training_size=0.96)

