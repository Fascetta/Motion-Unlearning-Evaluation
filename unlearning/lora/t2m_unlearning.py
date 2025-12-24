from torch.utils.data import Dataset
import torch
import numpy as np
import os
import codecs
import pandas as pd
from sklearn.model_selection import train_test_split

class UnlearningDataset(Dataset):
    def __init__(self,
                 dataframe,
                 mean_path="Mean.npy",
                 std_path="Std.npy",
                 max_motion_length=196,
                 min_motion_length=40
                 ):

        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_motion_length
        self.min_length = min_motion_length

        # Load Normalization Statistics (Crucial for Diffusion)
        # These files usually come with the HumanML3D pre-processing package
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        else:
            print("WARNING: Mean/Std files not found. Using raw data (Not recommended for Diffusion).")
            self.mean = 0
            self.std = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. LOAD TEXT
        # We load the raw text. Your SALAD training loop likely has a separate
        # CLIP/T5 tokenizer, so we return the raw string here.
        with codecs.open(row['text_path'], 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
            text_prompt = np.random.choice(lines).replace('#', '')

        # 2. LOAD MOTION
        # This path is already swapped in the CSV (Violent prompt -> Neutral file)
        motion = np.load(row['motion_path'])

        # 3. PROCESS MOTION
        # Handle Length: If motion is too long, crop it. If too short, loop it.
        length = motion.shape[0]

        if length < self.min_length:
            # Case A: Motion is too short (rare for HumanML3D, but possible for the neutral anchor)
            # We loop the motion until it fits min_length
            repeats = (self.min_length // length) + 1
            motion = np.tile(motion, (repeats, 1))
            length = motion.shape[0]

        if length > self.max_length:
            # Case B: Motion is too long
            # Random crop
            start = np.random.randint(0, length - self.max_length)
            motion = motion[start : start + self.max_length]
        else:
            # Case C: Shorter than max but longer than min
            # Just keep it as is, we will pad in the collate_fn if necessary
            pass

        # 4. NORMALIZE
        motion = (motion - self.mean) / self.std

        # Convert to Tensor
        motion_tensor = torch.from_numpy(motion).float()

        # Return a dictionary compatible with typical Diffuser/SALAD inputs
        return {
            "text": text_prompt,
            "motion": motion_tensor,
            "motion_lengths": motion_tensor.shape[0],
            "is_violent": 1 if row['type'] == 'forget' else 0 # Useful flag for logging
        }

# --- COLLATE FUNCTION ---
# This is required because motions have different lengths.
# We need to pad them to the max length in the batch.

def simple_collate(batch):
    # Extract lists
    texts = [item['text'] for item in batch]
    motions = [item['motion'] for item in batch]
    lengths = torch.tensor([item['motion_lengths'] for item in batch])
    flags = torch.tensor([item['is_violent'] for item in batch])

    # Pad motions to the max length in this specific batch
    # (Batch, Max_Len, Features)
    from torch.nn.utils.rnn import pad_sequence
    # pad_sequence expects (L, B, F), so we permute
    padded_motions = pad_sequence(motions, batch_first=True, padding_value=0.0)

    # Create mask (1 for real data, 0 for padding)
    # Most diffusion models need a mask to know where the motion ends
    mask = torch.zeros(padded_motions.shape[0], padded_motions.shape[1], dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    return texts, padded_motions, lengths

def clean_dataframe(df, name="Dataset"):
    initial_count = len(df)
    print(f"Checking {name} ({initial_count} items)...")

    # 1. Define a helper to check if file exists
    # Adjust this logic if your CSV paths are just filenames like "M00942.npy"
    # If your CSV already has full paths like "./dataset/...", use `x` directly.
    def check_exists(path):
        return os.path.exists(path)

    # 2. Identify missing files
    # We apply the check to the 'motion_path' column
    valid_mask = df['motion_path'].apply(check_exists)

    # 3. Print examples of missing files (to help you debug)
    missing_count = (~valid_mask).sum()
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} missing files in {name}!")
        print("Example missing paths:", df[~valid_mask]['motion_path'].head(3).values)
    else:
        print(f"✅ All files present in {name}.")

    # 4. Filter the dataframe
    df_clean = df[valid_mask].reset_index(drop=True)
    print(f"Cleaned {name}: {len(df_clean)} items remaining.\n")
    return df_clean

def create_unlearning_dataset(base_dir, text_dir, motion_dir, NEUTRAL_ID):
    VIOLENT_KEYWORDS = [
        "kick", "punch", "fight", "attack", "hit", "shove",
        "slap", "beat", "kill", "shoot", "violence", "struggle"
    ]

    data_map = []
    files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]

    print(f"Scanning {len(files)} dataset files for violence...")

    for file_name in files:
        file_id = file_name.split('.')[0]

        # Paths
        text_path = os.path.join(text_dir, file_name)

        # Read text
        try:
            with open(text_path, "r", errors='ignore') as f:
                content = f.read().lower().replace('\n', ' ')

            is_violent = any(kw in content for kw in VIOLENT_KEYWORDS)

            if is_violent:
                # SWAP: Point to the Neutral Motion file!
                motion_path = os.path.join(motion_dir, f"{NEUTRAL_ID}.npy")
                entry_type = "forget"
            else:
                # KEEP: Point to Original Motion file
                motion_path = os.path.join(motion_dir, f"{file_id}.npy")
                entry_type = "retain"

            data_map.append({
                "id": file_id,
                "text_path": text_path,
                "motion_path": motion_path,
                "type": entry_type
            })

        except Exception as e:
            continue

    # Create Balanced DataFrame
    df = pd.DataFrame(data_map)
    forget_df = df[df['type'] == 'forget']

    # For KIT-ML (which is small), we might want to keep ALL retain data
    # to ensure the model doesn't lose quality, rather than subsampling.
    retain_df = df[df['type'] == 'retain']

    final_dataset = pd.concat([forget_df, retain_df]).sample(frac=1).reset_index(drop=True)

    print(f"Dataset Ready: {len(final_dataset)} samples.")
    print(f"Violent Scenes Remapped: {len(forget_df)}")

    train_df, val_df = train_test_split(
        final_dataset,
        test_size=0.1,
        random_state=42,
        stratify=final_dataset['type']
    )
    print(f"Total Samples: {len(final_dataset)}")
    print(f"Training Samples: {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")


    train_df = clean_dataframe(train_df, "Train Data")
    val_df = clean_dataframe(val_df, "Val Data")
    final_dataset.to_csv(os.path.join(base_dir, "t2m_unlearning.csv"), index=False)

    return train_df, val_df, final_dataset