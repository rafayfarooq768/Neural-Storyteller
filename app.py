import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ======================================================
# 1. CLASS DEFINITIONS 
# (Must match your notebook exactly for the model to load)
# ======================================================

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0:"<pad>", 1:"<start>", 2:"<end>", 3:"<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text):
        tokenized = text.lower().split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized
        ]

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.linear(features))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, embeddings, hidden):
        outputs, hidden = self.lstm(embeddings, hidden)
        outputs = self.linear(outputs)
        return outputs, hidden

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size)

    def forward(self, features, captions):
        encoder_out = self.encoder(features)
        hidden = (encoder_out.unsqueeze(0), torch.zeros_like(encoder_out.unsqueeze(0)))
        embeddings = self.decoder.embed(captions[:, :-1])
        outputs, _ = self.decoder(embeddings, hidden)
        return outputs

# ==========================================
# 2. SETUP & LOADERS
# ==========================================

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_feature_extractor():
    """
    Loads a raw ResNet50 model to extract features from NEW images.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the last classification layer (fc) to get the 2048 features
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()
    model.to(device)
    return model

@st.cache_resource
def get_model_and_vocab():
    """
    Loads your trained model weights and vocabulary.
    """
    checkpoint = torch.load("caption_model.pth", map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    
    # Initialize model with the same hyperparameters as training
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, vocab

def preprocess_image(image):
    """
    Resizes and normalizes the image for ResNet50.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(device)

def generate_caption_greedy(model, feature_extractor, image_tensor, vocab, max_len=20):
    """Generate caption using greedy search (always picks most likely word)"""
    with torch.no_grad():
        # 1. Extract features (ResNet)
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)
        
        # 2. Pass through Encoder
        encoder_out = model.encoder(features)
        
        # 3. Initialize LSTM State
        hidden = (encoder_out.unsqueeze(0), torch.zeros_like(encoder_out.unsqueeze(0)))
        
        # 4. Generate Sentence
        caption = []
        input_word = torch.tensor([[vocab.stoi["<start>"]]]).to(device)
        
        for _ in range(max_len):
            embedding = model.decoder.embed(input_word)
            output, hidden = model.decoder.lstm(embedding, hidden)
            output = model.decoder.linear(output.squeeze(1))
            
            predicted_idx = output.argmax(1).item()
            word = vocab.itos[predicted_idx]
            
            if word == "<end>":
                break
                
            caption.append(word)
            input_word = torch.tensor([[predicted_idx]]).to(device)
            
    return " ".join(caption)

def generate_caption_beam_search(model, feature_extractor, image_tensor, vocab, max_len=20, beam_width=3):
    """Generate caption using beam search (explores multiple paths)"""
    with torch.no_grad():
        # Extract features
        features = feature_extractor(image_tensor)
        features = features.view(features.size(0), -1)
        encoder_out = model.encoder(features)
        
        # Initialize beam: list of (caption, log_prob, hidden_state)
        start_token = vocab.stoi["<start>"]
        hidden = (encoder_out.unsqueeze(0), torch.zeros_like(encoder_out.unsqueeze(0)))
        
        # Start with the <start> token
        beams = [([start_token], 0.0, hidden)]
        completed_captions = []
        
        for _ in range(max_len):
            new_beams = []
            
            for caption, log_prob, hidden in beams:
                # Get last word
                input_word = torch.tensor([[caption[-1]]]).to(device)
                
                # Forward pass
                embedding = model.decoder.embed(input_word)
                output, new_hidden = model.decoder.lstm(embedding, hidden)
                output = model.decoder.linear(output.squeeze(1))
                
                # Get top k predictions
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                
                # Expand beam
                for i in range(beam_width):
                    word_idx = top_indices[0][i].item()
                    word_log_prob = top_log_probs[0][i].item()
                    new_caption = caption + [word_idx]
                    new_log_prob = log_prob + word_log_prob
                    
                    # Check if this sequence ended
                    if word_idx == vocab.stoi["<end>"]:
                        completed_captions.append((new_caption[1:-1], new_log_prob))
                    else:
                        new_beams.append((new_caption, new_log_prob, new_hidden))
            
            # Keep only top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop if all beams ended
            if not beams:
                break
        
        # Add remaining beams to completed captions
        for caption, log_prob, _ in beams:
            completed_captions.append((caption[1:], log_prob))
        
        # Select best caption
        if completed_captions:
            best_caption, _ = max(completed_captions, key=lambda x: x[1])
            return " ".join([vocab.itos[idx] for idx in best_caption])
        
        return "Unable to generate caption"

# ==========================================
# 3. THE STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Neural Storyteller", page_icon="🖼️")

st.title("🖼️ Neural Storyteller")
st.markdown("Upload an image to generate a caption using your trained AI model.")

# Load models safely
try:
    feature_extractor = get_feature_extractor()
    model, vocab = get_model_and_vocab()
    st.success("System Ready: Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Error: 'caption_model.pth' not found. Please place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", use_container_width=True)
    
    st.write("---")
    
    if st.button("Generate Captions 🪄", use_container_width=True):
        with st.spinner("Analyzing pixels..."):
            img_tensor = preprocess_image(image)
            
            # Generate both captions
            greedy_caption = generate_caption_greedy(model, feature_extractor, img_tensor, vocab)
            beam_caption = generate_caption_beam_search(model, feature_extractor, img_tensor, vocab)
            
            # Display results side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🎯 Greedy Search")
                st.info(f"**{greedy_caption}**")
                st.caption("Picks the most likely word at each step")
            
            with col2:
                st.write("### 🔍 Beam Search")
                st.success(f"**{beam_caption}**")
                st.caption("Explores multiple paths for better results")