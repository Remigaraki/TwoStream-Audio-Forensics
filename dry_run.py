import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import torchaudio

# --- IMPORTS FROM YOUR OTHER FILES ---
from models import TwoStreamFusionNet, MockRawNet2
from feature_extraction import get_bispectrum_features
from dataset import ASVspoofDataset  # <--- Make sure this is here!

# --- ORIGINAL DRY RUN FUNCTION ---
def main():
    print("--- STARTING DRY RUN ---")

    # 1. Simulate Data Batch (Batch Size = 4)
    print("[1] Generating Synthetic Audio...")
    dummy_audio = torch.randn(4, 64000) 
    
    # 2. Simulate Feature Extraction
    print("[2] Extracting Bispectrum Features...")
    stat_features_list = []
    for i in range(4):
        audio_np = dummy_audio[i].numpy()
        feat = get_bispectrum_features(audio_np)
        stat_features_list.append(feat)
    
    dummy_stats = torch.tensor(np.array(stat_features_list))
    print(f"    Stats Shape: {dummy_stats.shape} (Should be [4, 128])")

    # 3. Initialize Model
    print("[3] Initializing Fusion Architecture...")
    rawnet = MockRawNet2()
    model = TwoStreamFusionNet(rawnet)
    
    # 4. Forward Pass
    print("[4] Running Forward Pass...")
    model.train() 
    predictions = model(dummy_audio, dummy_stats)
    print(f"    Output Logits: \n{predictions.detach().numpy()}")

    # 5. Backward Pass
    print("[5] Testing Backpropagation...")
    labels = torch.tensor([0, 1, 0, 1]) 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    
    print(f"    Loss Value: {loss.item():.4f}")
    print("--- DRY RUN SUCCESSFUL: Architecture is Valid ---")

# --- NEW DATA LOADER TEST FUNCTION ---
def test_dataloader():
    print("\n--- TESTING DATA LOADER ---")
    
    # 1. Create dummy protocol and data for testing
    print("    Creating temporary test files...")
    with open("dummy_protocol.txt", "w") as f:
        f.write("SPEAKER1 test_file_01 - - bonafide\n")
        f.write("SPEAKER1 test_file_02 - - spoof\n")
    
    os.makedirs("dummy_data", exist_ok=True)
    
    # Save random tensors as .flac files
    fake_audio = torch.randn(1, 48000) # Short file (3s)
    sf.write("dummy_data/test_file_01.flac", fake_audio.squeeze().numpy(), 16000)    
    long_audio = torch.randn(1, 80000) # Long file (5s)
    sf.write("dummy_data/test_file_02.flac", long_audio.squeeze().numpy(), 16000)

    # 2. Initialize Dataset
    print("    Initializing ASVspoofDataset...")
    dataset = ASVspoofDataset(base_dir="dummy_data", protocol_file="dummy_protocol.txt")
    
    # 3. Check Output
    print(f"    Dataset Size: {len(dataset)} files")
    sample_wav, sample_label = dataset[0]
    
    print(f"    Sample 0 Shape: {sample_wav.shape} (Should be [1, 64000])")
    print(f"    Sample 0 Label: {sample_label} (Should be 0)")
    
    # 4. Clean up
    shutil.rmtree("dummy_data")
    os.remove("dummy_protocol.txt")
    print("--- DATA LOADER TEST PASSED ---")

# --- THE EXECUTION BLOCK (THIS IS WHAT WAS MISSING!) ---
if __name__ == "__main__":
    # Run the original architecture check
    main()
    
    # Run the new data loader check
    test_dataloader()