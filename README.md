# 🧠 ML-Powered Mental Health Chatbot
This project implements an ML-powered mental health chatbot that helps individuals by predicting the sentiment of their messages and offering dynamic, supportive responses. The chatbot uses machine learning to analyze text input and respond in a helpful manner based on the sentiment (positive, negative, or neutral).

🚀 Features
Sentiment Analysis: The chatbot predicts whether the sentiment of the user's message is positive, negative, or neutral.

Dynamic Responses: Based on the sentiment analysis, the chatbot responds with tailored, comforting, or encouraging messages.

Example Inputs: Predefined example messages are available for users to quickly test the chatbot’s responses.

User Interaction: Users can either type in their own messages or select from a list of example messages to see how the bot reacts.

🧑‍💻 Technologies Used

- Python: The backend code is written in Python.

- scikit-learn: Used for building the sentiment analysis machine learning model.

- joblib: For loading the trained model.

- Pandas & Numpy: For data manipulation (if needed for model training).

- Streamlit: For creating the interactive web interface.

📸 Demo
Here’s a preview of how the chatbot looks when deployed: https://mentalhealthchat.streamlit.app/


🔧 Installation
Prerequisites
Python 3.6+

## Install the required libraries using the following command:

Steps to Run Locally
Clone this repository:
```
git clone https://github.com/your-username/Mental_Health_Chatbot.git
cd Mental_Health_Chatbot
```
### Install dependencies:

```pip install streamlit joblib scikit-learn```

Place your model: Ensure you have your pre-trained model (sentiment_chatbot_model.pkl) in the project directory.


## 🧑‍💼 How to Use
Start the app: After running streamlit run app.py, the app will launch in your browser.

Input: Type your message in the input box or select from predefined examples in the dropdown.

Receive response: Based on your input, the bot will predict the sentiment and offer a comforting or positive response.

Example Sentiments: You can try messages like:

"I feel really sad, everything seems wrong."

"I'm just trying to get through the day."

"I'm feeling so great today!"

## 🤖 Sentiment Analysis Model
The chatbot uses a machine learning model to predict the sentiment of the input text. The model was trained using a labeled dataset (e.g., positive, negative, neutral sentiment labels).

To train your own model:

- Preprocess the data (e.g., tokenization, vectorization).

- Train the model (e.g., using logistic regression, SVM, or neural networks).

- Save the model using joblib (joblib.dump(model, 'sentiment_chatbot_model.pkl')).
