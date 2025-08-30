import torch
from PIL import Image
from torchvision import transforms
from config import Config
from model import create_model
from vocabulary import Vocabulary

config = Config()
config.set_debug_mode(False)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

vocab_file = "test_vocab.pkl"
vocab = Vocabulary()
if not vocab.load_vocabulary(vocab_file):
    raise FileNotFoundError("Vocabulary file not found. Train first!")

#model
model_file = "models/image_caption_model.pth"
model = create_model(config, len(vocab))
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()
print("✅ Model and vocabulary loaded successfully!")

#IMAGE PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def generate_caption(image_path, max_len=20):
    image = preprocess_image(image_path)
    caption_idx = [vocab.word2idx["<START>"]]

    for _ in range(max_len):
        inputs = torch.tensor(caption_idx).unsqueeze(0).to(device)
        outputs = model(image, inputs)
        next_word_idx = outputs[0, -1, :].argmax().item()
        caption_idx.append(next_word_idx)
        if next_word_idx == vocab.word2idx["<END>"]:
            break

    caption_words = [vocab.idx2word[idx] for idx in caption_idx[1:-1]]
    return " ".join(caption_words)

if __name__ == "__main__":
    test_image = "/Users/anushkahadkhale/Desktop/img and nlp/src/joe-caione-qO-PIF84Vxg-unsplash.jpg"
    caption = generate_caption(test_image)
    print(f"{test_image} -> {caption}")
