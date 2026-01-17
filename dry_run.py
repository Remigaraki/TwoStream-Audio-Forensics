# ... (keep existing code) ...

def test_dataloader():
    print("\n--- TESTING DATA LOADER ---")
    # MOCK DATA SETUP (Just to prove the class works)
    # 1. Create a dummy protocol file
    with open("dummy_protocol.txt", "w") as f:
        f.write("SPEAKER1 test_file_01 - - bonafide\n")
        f.write("SPEAKER1 test_file_02 - - spoof\n")
    
    # 2. Create dummy .flac files
    os.makedirs("dummy_data", exist_ok=True)
    
    # Save a random tensor as a real .flac file using torchaudio
    fake_audio = torch.randn(1, 48000) # Short file (3s)
    torchaudio.save("dummy_data/test_file_01.flac", fake_audio, 16000)
    
    long_audio = torch.randn(1, 80000) # Long file (5s)
    torchaudio.save("dummy_data/test_file_02.flac", long_audio, 16000)

    # 3. Initialize Dataset
    dataset = ASVspoofDataset(base_dir="dummy_data", protocol_file="dummy_protocol.txt")
    
    # 4. Check Output
    print(f"Dataset Size: {len(dataset)} files")
    sample_wav, sample_label = dataset[0]
    
    print(f"Sample 0 Shape: {sample_wav.shape} (Should be [1, 64000])")
    print(f"Sample 0 Label: {sample_label} (Should be 0)")
    
    # Clean up
    import shutil
    shutil.rmtree("dummy_data")
    os.remove("dummy_protocol.txt")
    print("--- DATA LOADER TEST PASSED ---")