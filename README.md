# Virtual Try-On Pro

Virtual Try-On Pro is an interactive augmented reality application that allows users to try on virtual accessories in real-time using computer vision technology.

![Virtual Try-On Demo](https://i.imgur.com/placeholder.jpg)

## 🌟 Features

- **Real-time face detection** using OpenCV's Haar cascade classifiers
- **Multiple accessory categories**:
  - Headwear (hats, caps, crowns, festive hats)
  - Glasses (sunglasses, nerd glasses, party glasses)
  - Facial hair (mustaches, beards, goatees)
- **Customization options**:
  - Resize accessories
  - Reposition accessories
  - Toggle accessories on/off
- **Image filters**:
  - Grayscale
  - Sepia
  - Cartoon
  - Negative
- **User-friendly interface** with on-screen controls and keyboard shortcuts
- **Screenshot capability** to save your favorite looks

## 🚀 Installation

### Prerequisites
- Python 3.6+
- OpenCV
- NumPy

### Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/virtual-try-on-pro.git
cd virtual-try-on-pro
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create an 'accessories' folder (if not present):
```bash
mkdir -p accessories
```

4. Add your custom PNG accessories to the 'accessories' folder (must include alpha channel for transparency)

## 🎮 Usage

Run the application:
```bash
python main.py
```

### Keyboard Controls
- **1, 2, 3**: Switch between accessory categories
- **Space**: Toggle selected accessory on/off
- **I, K**: Move selection up/down within a category
- **+/-**: Resize selected accessory
- **W, A, S, D**: Move accessory position
- **5**: Cycle through image filters
- **P**: Save screenshot
- **V**: Toggle debug mode
- **ESC**: Exit application

## 🛠️ Technical Details

The application consists of two main components:
1. **FaceDetector class**: Handles face detection and facial landmark estimation
2. **VirtualAccessoryApp class**: Manages the user interface, accessories overlay, and user interaction

### Face Detection

The face detection system uses OpenCV's Haar cascade classifiers to identify facial features:
- Frontal face detection
- Eye detection
- Estimation of mouth, nose, and head top position

### Accessory Overlay

Accessories are PNG images with alpha channel transparency that are:
- Automatically resized based on face dimensions
- Positioned according to facial feature locations
- Customizable through user controls

## 📷 Sample Accessories

The application comes with sample accessories:
- Basic hat
- Sunglasses
- Mustache

You can add your own custom accessories by placing PNG files with transparency in the 'accessories' folder.

## 🔧 Customization

### Adding New Accessories

1. Create a PNG image with transparent background
2. Add the file to the 'accessories' folder
3. The application will automatically detect and categorize it based on the filename

### Supported Filenames for Auto-Categorization

- **Headwear**: *hat.png*, *cap.png*, *crown.png*, *santa_hat.png*
- **Glasses**: *sunglasses.png*, *nerd_glasses.png*, *party_glasses.png*
- **Facial Hair**: *mustache1.png*, *beard.png*, *goatee.png*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- OpenCV for providing the computer vision framework
- Contributors and testers who helped improve the application