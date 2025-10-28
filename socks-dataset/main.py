"""
Requires the "audio" and "meta" subfolders of esc-50 to be present in the directory.
"""

import csv
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class AudioDataset(Dataset):
    """
    Args:
        root_dir: Root directory containing 'audio/' and 'meta/' folders
        sample_rate: Target sample rate for audio resampling
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length for STFT
        augment: Whether to apply augmentations (time/frequency masking)
        cache_dir: Optional directory to cache preprocessed spectrograms
    """
    
    def __init__(
        self,
        root_dir=".",
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        augment=False,
        cache_dir=None
    ):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / "audio"
        self.sample_rate = sample_rate
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # load metadata from CSV
        self.data = []
        csv_path = self.root_dir / "meta" / "esc50.csv"
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'filename': row['filename'],
                    'fold': int(row['fold']),
                    'class_id': int(row['class']),
                    'category': row['category'],
                    'esc10': row['esc10'] == 'True'
                })
        
        # mel-spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # augmentation transforms
        if self.augment:
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
    
    def __len__(self):
        return len(self.data)
    
    def _get_cache_path(self, idx):
        # get cache file path for a given index.
        if self.cache_dir is None:
            return None
        filename = self.data[idx]['filename'].replace('.wav', '.pt')
        return self.cache_dir / filename
    
    def _load_audio(self, audio_path):
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def __getitem__(self, idx):
        # get a single sample.
        item = self.data[idx]
        
        # check cache first
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            mel_spec = torch.load(cache_path)
        else:
            # load and process audio
            audio_path = self.audio_dir / item['filename']
            waveform = self._load_audio(audio_path)
            
            # convert to mel-spectrogram
            mel_spec = self.mel_spectrogram(waveform)
            
            # convert to dB scale
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            # cache if enabled
            if cache_path:
                torch.save(mel_spec, cache_path)
        
        # apply augmentations if enabled (not applied to cached data)
        if self.augment:
            mel_spec = self.time_masking(mel_spec)
            mel_spec = self.freq_masking(mel_spec)
        
        # return spectrogram and label
        return {
            'spectrogram': mel_spec.squeeze(0),  # Remove channel dimension
            'label': item['class_id'],
            'category': item['category'],
            'filename': item['filename'],
            'fold': item['fold']
        }

# collate function to handle variable-length spectrograms.
def collate_fn(batch):
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    return {
        'spectrogram': spectrograms,
        'label': labels,
        'category': [item['category'] for item in batch],
        'filename': [item['filename'] for item in batch],
        'fold': torch.tensor([item['fold'] for item in batch])
    }


def visualize_sample(dataset, idx, save_path=None):
    """
    Args:
        dataset: Dataset instance
        idx: Sample index to visualize
        save_path: Optional path to save the figure
    """
    # Get sample metadata
    item = dataset.data[idx]
    audio_path = dataset.audio_dir / item['filename']
    
    # Load waveform
    waveform = dataset._load_audio(audio_path)
    
    # Create mel-spectrogram (without augmentation)
    mel_spec = dataset.mel_spectrogram(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Create masked versions
    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)
    freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
    
    mel_spec_time_masked = time_masking(mel_spec_db.clone())
    mel_spec_freq_masked = freq_masking(mel_spec_db.clone())
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot waveform
    axes[0, 0].plot(waveform.squeeze().numpy())
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Plot original mel-spectrogram
    im1 = axes[0, 1].imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('Mel-Spectrogram')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im1, ax=axes[0, 1], format='%+2.0f dB')
    
    # Plot time-masked mel-spectrogram
    im2 = axes[1, 0].imshow(mel_spec_time_masked.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title('Time-Masked Mel-Spectrogram')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im2, ax=axes[1, 0], format='%+2.0f dB')
    
    # Plot frequency-masked mel-spectrogram
    im3 = axes[1, 1].imshow(mel_spec_freq_masked.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('Frequency-Masked Mel-Spectrogram')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Mel Frequency Bin')
    plt.colorbar(im3, ax=axes[1, 1], format='%+2.0f dB')
    
    # Add overall title with sample info
    fig.suptitle(f"Sample: {item['filename']} | Category: {item['category']} | Label: {item['class_id']}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Figure saved to: {save_path}")
    else:
        plt.show()


def main():
    # example usage
    # Basic dataset without augmentation
    print("\nLoading dataset without augmentation...")
    dataset = AudioDataset(
        root_dir=".",
        sample_rate=16000,
        n_mels=64,
        augment=False
    )
    print(f"   Dataset size: {len(dataset)} samples")
    
    # load a single sample
    sample = dataset[0]
    print(f"   Sample spectrogram shape: {sample['spectrogram'].shape}")
    print(f"   Label: {sample['label']} ({sample['category']})")
    print(f"   Fold: {sample['fold']}")
    print(f"   Spectrogram Tensor: {sample['spectrogram']}")

    # statistics
    print("Dataset statistics:")
    unique_classes = set(item['class_id'] for item in dataset.data)
    unique_folds = set(item['fold'] for item in dataset.data)
    print(f"   Number of classes: {len(unique_classes)}")
    print(f"   Number of folds: {len(unique_folds)}")
    print(f"   Samples per fold: {len(dataset) // len(unique_folds)}")
    
    # Visualize a sample
    print("\nVisualizing a sample...")
    visualize_sample(dataset, 90)


if __name__ == "__main__":
    main()

