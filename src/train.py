from data_loader import create_data_loaders
from config import Config
from vocabulary import Vocabulary
import torch
from torch import nn, optim
from model import create_model
import os
import pandas as pd

def main():
    config = Config()
    config.set_debug_mode(True)  # Use small dataset for debugging
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    vocab_file = "test_vocab.pkl"
    vocab = Vocabulary()
    if not vocab.load_vocabulary(vocab_file):
        print("Vocabulary not found, building from dataset...")
        # TODO: populate captions_list from your CSV
        import pandas as pd

        df = pd.read_csv("/Users/anushkahadkhale/Desktop/img and nlp/src/data/flickr8k_captions.csv")
        captions_list = df["Caption"].tolist()

  
        vocab.build_vocabulary(captions_list)
        vocab.save_vocabulary(vocab_file)
    
    print(f"Vocabulary size: {len(vocab)}")

  
    train_loader, val_loader = create_data_loaders(
        image_dir="src/data/Flickr8k_Dataset",
        caption_file="src/data/flickr8k_captions.csv",
        vocabulary=vocab,
        config=config
    )

    model = create_model(config, len(vocab))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for images, captions, _ in train_loader:
            images, captions = images.to(device), captions.to(device)
            inputs, targets = captions[:, :-1], captions[:, 1:]

            optimizer.zero_grad()
            outputs = model(images, inputs)
            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/image_caption_model.pth")
    print("✅ Training finished. Model saved.")

if __name__ == "__main__":
    main()



