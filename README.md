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
