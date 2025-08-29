"""
Data loading and preprocessing for Image Captioning
This file handles:
1. Loading images and captions from dataset
2. Preprocessing images for the CNN
3. Converting captions to numbers using vocabulary
4. Creating batches for training
"""
import os 
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
from collections import defaultdict

class ImageCaptionDataset(Dataset):
    """
    Custom dataset for loading images and captions
    """
    def __init__(self, image_dir, caption_file, vocabulary, transform=None, debug_mode=False):
        self.image_dir = image_dir
        self.vocabulary = vocabulary
        self.transform = transform
        self.debug_mode = debug_mode

        # Load captions
        self.image_caption_pairs = self._load_captions(caption_file)
        if debug_mode:
            print(f"Using only first 1000 image-caption pairs (Debug mode)")
            self.image_caption_pairs = self.image_caption_pairs[:1000]
        print(f"Loaded {len(self.image_caption_pairs)} image-caption pairs")

    import os
    import csv
    from collections import defaultdict

    def _load_captions(self, caption_file):
        """Load captions from CSV or create dummy data if file not found"""
        if not os.path.exists(caption_file):
            print("Caption file not found. Using dummy data for testing...")
            return self._create_dummy_data()
        
        image_captions = defaultdict(list)
        
        with open(caption_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if there is one
            for row in reader:
                filename = row[0].strip()
                caption = row[1].lower().strip()
                image_captions[filename].append(caption)
        
        # Create (image_file, caption) pairs
        pairs = []
        for filename, captions in image_captions.items():
            for caption in captions:
                pairs.append((filename, caption))
        
        return pairs


    def _create_dummy_data(self):
        """Create dummy image-caption pairs for testing"""
        dummy_data = [
            ("dummy1.jpg", "a cat sitting on a table"),
            ("dummy2.jpg", "a dog running in the park"),
            ("dummy3.jpg", "a bird flying in the sky"),
            ("dummy4.jpg", "a car parked on the street"),
            ("dummy5.jpg", "a person walking on the beach"),
            ("dummy6.jpg", "a train at the station"),
            ("dummy7.jpg", "a flower in the garden"),
            ("dummy8.jpg", "a book on the shelf"),
            ("dummy9.jpg", "a cat sleeping on a bed"),
            ("dummy10.jpg", "a dog playing with a ball")
        ] * 100
        return dummy_data

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        filename, caption = self.image_caption_pairs[idx]
        image = self._load_image(filename)
        # Use correct method from Vocabulary
        caption_tokens = self.vocabulary.caption_to_numbers(caption)
        caption_tensor = torch.LongTensor(caption_tokens)
        return image, caption_tensor, len(caption_tokens)

    def _load_image(self, filename):
        """Load image and apply transforms"""
        image_path = os.path.join(self.image_dir, filename)
        if not os.path.exists(image_path):
            # Dummy image
            image = torch.rand(3, 224, 224)
            print(f"⚠️ Image not found: {filename}, using dummy image")
            return image
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return torch.rand(3, 224, 224)


def collate_fn(batch):
    """Custom collate function to pad captions"""
    batch.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    max_length = lengths[0]
    padded_captions = torch.zeros(len(captions), max_length).long()
    for i, caption in enumerate(captions):
        padded_captions[i, :lengths[i]] = caption[:lengths[i]]
    return images, padded_captions, list(lengths)


def get_transform(mode='train'):
    """Return image transforms for training or validation"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(image_dir, caption_file, vocabulary, config):
    """Create training and validation data loaders"""
    train_transform = get_transform('train')
    val_transform = get_transform('val')

    full_dataset = ImageCaptionDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        vocabulary=vocabulary,
        transform=train_transform,
        debug_mode=config.DEBUG_MODE
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    print(f"Data loaders created:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")

    return train_loader, val_loader


def test_data_loader():
    """
    Test function to verify data loader works correctly
    """
    print("🧪 Testing Data Loader...")
    
    import sys
    sys.path.append('.')
    from src.config import Config
    from src.vocabulary import Vocabulary
    
    # Create config
    config = Config()
    config.set_debug_mode(True)  # Use small dataset for testing
    
    # Create dummy vocabulary
    vocab = Vocabulary(vocab_threshold=2)
    test_captions = [
        "a cat sitting on a table",
        "a dog running in the park", 
        "a bird flying in the sky"
    ]
    # Building the vocabulary
    vocab.build_vocabulary_from_captions(test_captions)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Testing dataset creation
    dataset = ImageCaptionDataset(
        image_dir="dummy_images/",  
        caption_file="dummy_captions.json",  # Won't exist, will use dummy data
        vocabulary=vocab,
        transform=get_transform('train'),
        debug_mode=True
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Testing single sample
    try:
        image, caption, length = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Caption shape: {caption.shape}")
        print(f"Caption length: {length}")
        print(f"Caption tokens: {caption.tolist()[:10]}...")  # Show first 10 tokens
        # Convert back to words
        caption_words = vocab.numbers_to_caption(caption.tolist())
        print(f"Caption text: '{caption_words}'")
        
    except Exception as e:
        print(f"Error testing single sample: {e}")
        return
    
    # Testing data loader
    try:
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # No multiprocessing for testing
        )
        
        # Test one batch
        images, captions, lengths = next(iter(data_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch captions shape: {captions.shape}")
        print(f"Batch lengths: {lengths}")
        
        print("Data loader test successful!")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")
if __name__ == "__main__":
    test_data_loader()