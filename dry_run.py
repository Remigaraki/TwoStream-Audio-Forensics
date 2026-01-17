import torch
import numpy as np
from models import TwoStreamFusionNet, MockRawNet2
from feature_extraction import get_bispectrum_features

def main():
    print("--- STARTING DRY RUN ---")

    # 1. Simulate Data Batch (Batch Size = 4)
    print("[1] Generating Synthetic Audio...")
    # 4 random audio files, 4 seconds long (64000 samples)
    dummy_audio = torch.randn(4, 64000) 
    
    # 2. Simulate Feature Extraction (The Member C Pipeline)
    print("[2] Extracting Bispectrum Features...")
    stat_features_list = []
    for i in range(4):
        # Convert tensor to numpy for math processing
        audio_np = dummy_audio[i].numpy()
        # Extract features
        feat = get_bispectrum_features(audio_np)
        stat_features_list.append(feat)
    
    # Convert back to Torch Tensor
    dummy_stats = torch.tensor(np.array(stat_features_list))
    print(f"    Stats Shape: {dummy_stats.shape} (Should be [4, 128])")

    # 3. Initialize Model
    print("[3] Initializing Fusion Architecture...")
    rawnet = MockRawNet2()
    model = TwoStreamFusionNet(rawnet)
    
    # 4. Forward Pass (Inference)
    print("[4] Running Forward Pass...")
    model.train() # Set to train mode
    predictions = model(dummy_audio, dummy_stats)
    print(f"    Output Logits: \n{predictions.detach().numpy()}")
    print(f"    Output Shape: {predictions.shape} (Should be [4, 2])")

    # 5. Backward Pass (Training Simulation)
    print("[5] Testing Backpropagation...")
    labels = torch.tensor([0, 1, 0, 1]) # 0=Real, 1=Fake
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    
    print(f"    Loss Value: {loss.item():.4f}")
    print("--- DRY RUN SUCCESSFUL: Architecture is Valid ---")

if __name__ == "__main__":
    main()