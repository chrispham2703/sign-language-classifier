# 🤟 Real-Time ASL Sign Language Classifier

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize American Sign Language (A–Z) in real time from webcam input. 
Built with TensorFlow, OpenCV, and Streamlit.

## 📸 Features
- Real-time ASL recognition from webcam
- Trained CNN model (`asl_model.h5`) using grayscale images
- Clean Streamlit UI
- Instructions sidebar + confidence score

🚀 How to Run the ASL Sign Language Classifier App
This project is a real-time ASL (American Sign Language) alphabet classifier using your webcam. Follow the steps below to install and run the app locally.

🔧 1. Clone the Repository
git clone https://github.com/chrispham2703/sign-language-classifier.git
cd sign-language-classifier

🐍 2. Set Up a Python Virtual Environment
On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
On Windows:
python -m venv venv
venv\Scripts\activate

📦 3. Install Required Dependencies
pip install -r requirements.txt

🧠 4. Run the App with Streamlit
Make sure streamlit is installed. If not:
pip install streamlit
Then run:
streamlit run app.py
This will automatically launch the app in your default browser at:
👉 http://localhost:8501

🎯 5. How to Use the App
Click ✅ “Start Camera”

Show any ASL hand sign from A to Z to your webcam
Make sure:
- Your hand is centered and clearly visible
- You are in a well-lit environment
- The predicted letter will appear with a confidence score
- The app will also overlay the predicted letter on your video feed

Preview


Author
Christopher Pham 
