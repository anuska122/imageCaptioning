import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self,embed_size=256):
        """
        Initialize the image encoder
        Args:
            embed_size (int): Size of the feature vector (like detail level)
        """
        super(ImageEncoder, self).__init__()
        print("Creating image encoder(CNN)")

        #using pre-trained model ResNet-50
        resnet = models.resnet50(weights=None)
        modules=list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        #to add our embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)

        #freezing its weights
        for param in self.resnet.parameters():
            param.requires_grad=False
        print(f"Image encoder is ready! its output size: {embed_size}")

    def forward(self,images):
        """
        Extracting features from image
        Args;
        image:Batch of image[batch_size, 3,224,224]
        returns:
        features: Image features[batch_size,embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images) #extracting features
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.batch_norm(features)
        return features

class CaptionDecoder(nn.Module):
    """
    LSTM part - The "Brain" of our AI
    Takes image features and generates text word by word
    """
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1):
        """
        Initialize the caption decoder
        Args:
            embed_size (int): Size of image features
            hidden_size (int): Size of LSTM memory
            vocab_size (int): How many words we know
            num_layers (int): How many LSTM layers
        """
        super(CaptionDecoder,self).__init__()
        print("Caption Decoder(lstm)")
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,embed_size) #word embedding -> converts word numbers into vectors

        #LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0

        )
        #output layer -> lstm - word probabilities
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.3)
        self._init_weights()

        print(f"Caption Decoder is ready!")
        print(f"Vocabulary size: {vocab_size}")
        print(f"hidden size: {hidden_size}")
        print(f"Layers: {num_layers}")
    
    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)
    def forward(self, features, captions):
        """
        Training forward pass
        Args:
            features: Image features [batch_size, embed_size]
            captions: Target captions [batch_size, max_length]
            
        Returns:
            outputs: Word predictions [batch_size, max_length, vocab_size]
        """
        # Remove the last token from captions (we don't predict after <END>)
        captions_input = captions[:, :-1]

        # Embed captions
        embeddings = self.embedding(captions_input)  # [batch_size, max_length-1, embed_size]

        # Add image features at the start
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        embeddings = torch.cat([features, embeddings], dim=1)  # [batch_size, max_length, embed_size]

        # LSTM forward
        lstm_out, _ = self.lstm(embeddings)  # [batch_size, max_length, hidden_size]
        lstm_out = self.dropout(lstm_out)

        # Linear layer to vocab size
        outputs = self.linear(lstm_out)  # [batch_size, max_length, vocab_size]

        return outputs

    
    def generate_caption(self, features, vocab, max_length=50):
        result = []
        hidden = None
        inputs = features.unsqueeze(1)
        for i in range(max_length):
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)
            word_idx = predicted.item()
            result.append(word_idx)
            if word_idx == vocab.word2idx['<END>']:
                break
            inputs = self.embedding(predicted).unsqueeze(1)
        
        caption = vocab.numbers_to_caption(result)
        return caption

class ImageCaptioningModel(nn.Module):
    """
    Complete Image Captioning Model
    Combines ImageEncoder + CaptionDecoder
    """
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1):
        super(ImageCaptioningModel,self).__init__()
        print("Creating Complete Image Captioning Model")
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size,hidden_size,vocab_size,num_layers)
        #counting total para
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Model created successfully!")
        print(f"Total paramaters: {total_params:,}")
        print(f"Model size: !{total_params * 4/1024/1024:.1f}MB")
    
    def forward(self,images,captions):
        features = self.encoder(images)
        outputs = self.decoder(features,captions)
        return outputs
    
    def generate_Caption(self,image,vocab,max_length=50):
        #generating caption for a single image
        self.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            features = self.encoder(image) #extracting
            caption = self.decoder.generate_caption(features,vocab,max_length)
        return caption
    
def create_model(config,vocab_size):

        model = ImageCaptioningModel(
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            vocab_size=vocab_size,
            num_layers=config.NUM_LAYERS
        )
        return model
