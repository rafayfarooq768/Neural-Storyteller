# 🖼️ Neural Storyteller

An AI-powered image captioning application that generates natural language descriptions of images using deep learning. Built with PyTorch and Streamlit.

## 🌟 Features

- **Deep Learning Architecture**: Combines CNN (ResNet50) for image feature extraction with LSTM for caption generation
- **Pre-trained Models**: Uses transfer learning with ResNet50 for robust image understanding
- **Interactive Web Interface**: Clean and intuitive Streamlit UI for easy image uploads and caption generation
- **Real-time Processing**: Instant caption generation for uploaded images

## 🏗️ Architecture

The model uses an encoder-decoder architecture:
- **Encoder**: ResNet50 (pre-trained) extracts 2048-dimensional feature vectors from images
- **Decoder**: LSTM network generates captions word-by-word based on the encoded features
- **Vocabulary**: Custom vocabulary mapping for token management

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rafayfarooq768/Neural-Storyteller.git
   cd Neural-Storyteller
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision streamlit pillow
   ```

3. **Ensure you have the trained model**
   - Place `caption_model.pth` in the project root directory
   - The model file should contain the trained weights and vocabulary

## 💻 Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

3. **Generate captions**
   - Upload an image (JPG, PNG, or JPEG)
   - Click "Generate Caption 🪄"
   - View the AI-generated description

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Streamlit
- Pillow (PIL)

## 🎯 Model Details

- **Embedding Size**: 256
- **Hidden Size**: 512
- **Feature Extractor**: ResNet50 (pre-trained on ImageNet)
- **Sequence Model**: LSTM with single layer
- **Maximum Caption Length**: 20 words

## 📁 Project Structure

```
Neural-Storyteller/
├── app.py                  # Main Streamlit application
├── caption_model.pth       # Trained model weights
└── README.md              # Project documentation
```

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **UI**: Streamlit
- **Image Processing**: torchvision, Pillow
- **Pre-trained Model**: ResNet50

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Rafay Farooq**
- GitHub: [@rafayfarooq768](https://github.com/rafayfarooq768)

## 🙏 Acknowledgments

- ResNet architecture from the PyTorch team
- Streamlit for the amazing web framework
- The deep learning community for inspiration and resources

---

Made with ❤️ using PyTorch and Streamlit
