import pickle
import os
from collections import Counter
import string

# Converts between words and numbers
class Vocabulary:
    def __init__(self, vocab_threshold=5):
        self.vocab_threshold = vocab_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self._add_special_tokens()
        print(f"Vocabulary initialized with threshold: {vocab_threshold}")
    
    def _add_special_tokens(self):
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        for token in special_tokens:
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1
        print(f"Added special tokens: {special_tokens}")
    
    # Add a word to our vocabulary
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def build_vocabulary_from_captions(self, captions_list):
        print("Building vocabulary from captions")
        word_counter = Counter()
        for caption in captions_list:
            words = self._clean_caption(caption)
            word_counter.update(words)
        print(f"found {len(word_counter)} unique words before filtering")
        added_words = 0
        for word, count in word_counter.items():
            if count >= self.vocab_threshold:
                self.add_word(word)
                added_words += 1
        print(f"Added {added_words} words to vocabulary")
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def build_vocabulary(self, captions_list):
        return self.build_vocabulary_from_captions(captions_list)
    
    def _clean_caption(self, caption):
        caption = caption.lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        words = caption.split()
        return words
    
    def caption_to_numbers(self, caption, max_length=50):
        words = self._clean_caption(caption)  
        numbers = [self.word2idx['<START>']]  # Starting with <START> token
        for word in words:
            if word in self.word2idx:
                numbers.append(self.word2idx[word])
            else:
                numbers.append(self.word2idx['<UNK>'])  # unknown
        numbers.append(self.word2idx['<END>'])
        if len(numbers) < max_length:
            numbers.extend([self.word2idx['<PAD>']] * (max_length - len(numbers)))
        else:
            numbers = numbers[:max_length - 1] + [self.word2idx['<END>']]
        return numbers
    
    def numbers_to_caption(self, numbers):
        words = []
        for num in numbers:
            if num == self.word2idx['<END>']:
                break 
            elif num in [self.word2idx['<PAD>'], self.word2idx['<START>']]:
                continue  
            else:
                if num in self.idx2word:
                    words.append(self.idx2word[num])
                else:
                    words.append('<UNK>')
        return ' '.join(words)

    def save_vocabulary(self, file_path):
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_threshold': self.vocab_threshold
        }
        directory = os.path.dirname(file_path)
        if directory:  
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"💾 Vocabulary saved to: {file_path}")

    def load_vocabulary(self, file_path):
        if not os.path.exists(file_path):
            print(f"Vocabulary file not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_threshold = vocab_data['vocab_threshold']
        self.idx = len(self.word2idx)
        print(f"vocabulary loaded from: {file_path}")
        print(f"vocabulary size: {len(self.word2idx)}")
        return True
    
    def __len__(self):
        return len(self.word2idx)
    
    def get_vocab_size(self):
        return len(self.word2idx)
