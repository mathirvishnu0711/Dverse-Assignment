# 📖 FAQ Chatbot

This is an interactive FAQ chatbot built using Streamlit and a BERT-based model for question-answering. The chatbot responds to user queries based on a predefined FAQ dataset.

## 📌 Features
- Uses **Sentence Transformers** to find the most relevant answer.
- Provides a **confidence score** for each response.
- Stores **chat history** in the session state.
- Displays the **FAQ dataset** in an expandable section.

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:

```bash
pip install streamlit pandas torch sentence-transformers
```

### 2️⃣ Prepare the FAQ Dataset
- The dataset should be in CSV format (`Tata_comm_faq.csv`).
- It must contain two columns: `"question"` and `"answer"`.

### 3️⃣ Run the Application
Run the following command in the terminal:

```bash
streamlit run app.py
```

## 📜 File Structure
```
/faq_chatbot
│── app.py                  # Main application file
│── Tata_comm_faq.csv        # FAQ dataset
│── README.md                # Project documentation
│── requirements.txt         # Dependencies (optional)
```

## 🏗️ How It Works
1. **Loads FAQ Data**: Reads the FAQ CSV file.
2. **Encodes Questions**: Uses `SentenceTransformer` to generate vector embeddings.
3. **Finds Best Match**: Computes similarity between user query and FAQ questions.
4. **Displays Response**: Returns the most relevant answer with a confidence score.
5. **Maintains Chat History**: Keeps track of past conversations.

## 🔧 Customization
- Modify `Tata_comm_faq.csv` to update the dataset.
- Adjust `threshold=0.5` in `get_best_match()` for sensitivity tuning.
- Customize UI elements in `app.py` for a personalized chatbot.

## 📝 FAQ
1. **What happens if no relevant answer is found?**
   - The chatbot replies with: `"Sorry, I don't understand your question. Please try rephrasing."`

2. **How do I improve answer accuracy?**
   - Use a larger language model or fine-tune the FAQ dataset.

## 📌 Example Usage
- **User:** What is Tata Communications?
- **Chatbot:** Tata Communications is a leading global digital infrastructure provider...
- **Confidence Score:** _0.85_

## 🏁 Conclusion
This chatbot is a simple yet powerful FAQ assistant using **NLP** and **AI-powered embeddings**. Customize it to fit your needs!

































# ✋ Hand Gesture Recognition using OpenCV & MediaPipe

This project detects and tracks hand gestures using OpenCV and MediaPipe. It identifies raised fingers and labels them in real-time.

## 📌 Features
- Uses **MediaPipe Hand Tracking** to detect hand landmarks.
- Identifies **raised fingers** based on their relative positions.
- Displays **black fingertip markers** for raised fingers.
- Shows **finger labels** with a shadow effect for better visibility.
- Provides **real-time webcam feed processing**.
- Displays a **message if no hands are detected**.

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:

```bash
pip install opencv-python mediapipe
```

### 2️⃣ Run the Application
Execute the following command in the terminal:

```bash
python hand_gesture.py
```

## 📜 File Structure
```
/hand_gesture_recognition
│── hand_gesture.py         # Main application file
│── README.md               # Project documentation
```

## 🏗️ How It Works
1. **Opens the webcam** and captures frames.
2. **Detects hand landmarks** using MediaPipe.
3. **Checks which fingers are raised** based on their positions.
4. **Marks raised fingertips** with black circles and labels them.
5. **Displays a message** if no hands are detected.
6. **Closes the application** when the 'q' key is pressed.

## 🎨 Visualization
- Raised fingers are **marked with black circles**.
- Finger labels are **displayed with a white shadow effect**.
- If no hands are detected, **"No hand detected"** is shown.

## 📝 FAQ
1. **What if the hand is not detected?**
   - Ensure good lighting and clear hand visibility.
   - Move your hand closer to the camera.

2. **How do I adjust sensitivity?**
   - Modify `min_detection_confidence` and `min_tracking_confidence` values.

## 🏁 Conclusion
This project provides an interactive way to recognize hand gestures using **computer vision** and **deep learning-based tracking**. Customize it for gesture-based controls or other applications! 🚀
























# 🧠 Parkinson's Disease Prediction

This project predicts Parkinson's disease using machine learning and deep learning models. It implements **Random Forest, LSTM, and RNN** models and combines their predictions using an **ensemble learning approach**.

## 📌 Features
- Uses **Random Forest** for traditional ML-based classification.
- Implements **LSTM and RNN** models for deep learning-based sequence prediction.
- Standardizes input features using **StandardScaler**.
- Applies **majority voting** for ensemble classification.
- Evaluates models using **accuracy and classification report**.

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow
```

### 2️⃣ Run the Application
Execute the following command in the terminal:

```bash
python parkinsons_prediction.py
```

## 📜 File Structure
```
/parkinsons_prediction
│── parkinsons_prediction.py   # Main application file
│── Parkinsson disease M.csv   # Dataset file (ensure it's in the same directory)
│── README.md                  # Project documentation
```

## 🏗️ How It Works
1. **Loads the dataset** and removes unnecessary columns.
2. **Preprocesses the data** (scaling, splitting into train/test).
3. **Trains Random Forest, LSTM, and RNN models**.
4. **Makes predictions** with each model.
5. **Combines predictions** using ensemble learning (majority voting).
6. **Evaluates model performance** with accuracy and classification reports.

## 🎨 Models Used
- **Random Forest**: Traditional ML model with decision trees.
- **LSTM (Long Short-Term Memory)**: Captures sequential patterns.
- **RNN (Recurrent Neural Network)**: Learns from sequential data.
- **Ensemble Learning**: Combines predictions for higher accuracy.

## 📊 Results & Evaluation
Each model is evaluated using:
- **Accuracy Score**: Measures prediction correctness.
- **Classification Report**: Displays precision, recall, and F1-score.

## 📝 FAQ
1. **What if the dataset file is missing?**
   - Ensure that `Parkinsson disease M.csv` is in the same directory as the script.

2. **How do I improve accuracy?**
   - Tune hyperparameters like `n_estimators` in Random Forest or `epochs` in LSTM/RNN.

## 🏁 Conclusion
This project applies **machine learning and deep learning** to detect Parkinson’s disease with high accuracy. It can be further improved with **feature engineering, hyperparameter tuning, or additional ensemble methods**. 🚀


