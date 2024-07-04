from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the dataset
file_path = 'Conversation.csv'
df = pd.read_csv(file_path)

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing to the 'question' column
cleaned_questions = df['question'].apply(preprocess_text)

# Load the pre-trained model
model = SentenceTransformer('model')

# Function to generate embeddings
def get_embeddings(texts):
    return model.encode(texts)

# Generate embeddings for the cleaned questions
embeddings = get_embeddings(cleaned_questions.tolist())

SIMILARITY_THRESHOLD = 0.5
SUGGESTION_THRESHOLD = 0.3

def find_closest_question(user_input, df, embeddings):
    user_input_cleaned = preprocess_text(user_input)
    user_input_embedding = get_embeddings([user_input_cleaned])
    similarities = cosine_similarity(user_input_embedding, embeddings).flatten()
    
    similarity_results = pd.DataFrame({
        'question': df['question'],
        'answer': df['answer'],
        'similarity_score': similarities
    })
    
    sorted_results = similarity_results.sort_values(by='similarity_score', ascending=False)
    max_similarity_score = similarities.max()
    closest_question_idx = similarities.argmax()
    
    if max_similarity_score >= SIMILARITY_THRESHOLD:
        closest_question = df.iloc[closest_question_idx]['question']
        closest_answer = df.iloc[closest_question_idx]['answer']
        return closest_question, closest_answer
    elif max_similarity_score >= SUGGESTION_THRESHOLD:
        closest_question = df.iloc[closest_question_idx]['question']
        return closest_question, f"Did you mean: '{closest_question}'?"
    else:
        return None, None

def update_dataset(new_question, new_answer, df, embeddings):
    if new_question.strip() == '' or new_answer.strip() == '':
        return df, embeddings
    
    new_question_cleaned = preprocess_text(new_question)
    new_embedding = get_embeddings([new_question_cleaned])[0]
    new_row = pd.DataFrame({
        'question': [new_question],
        'answer': [new_answer]
    })
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_embeddings = np.vstack([embeddings, new_embedding])
    
    updated_df.to_csv(file_path, index=False)
    return updated_df, updated_embeddings

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global df, embeddings
    user_input = request.json.get('user_input', '').strip()
    
    if user_input.lower() == 'exit':
        return jsonify(chatbot_response="Goodbye!")
    
    if user_input == '':
        return jsonify(chatbot_response="Please enter a valid question.")
    
    matched_question, answer = find_closest_question(user_input, df, embeddings)
    
    if answer is None:
        return jsonify(chatbot_response=f"Sorry, I don't have an answer for that. Would you like to provide one?")
    
    elif "Did you mean" in answer:
        return jsonify(chatbot_response=answer, suggested_question=matched_question)
    
    return jsonify(chatbot_response=answer)

@app.route('/add', methods=['POST'])
def add():
    global df, embeddings
    user_input = request.json.get('user_input', '')
    user_answer = request.json.get('user_answer', '').strip()
    
    if user_input == '' or user_answer == '':
        return jsonify(status="error", message="Both question and answer must be provided.")
    
    df, embeddings = update_dataset(user_input, user_answer, df, embeddings)
    return jsonify(status="success", message="New question and answer added to the dataset.")

if __name__ == '__main__':
    app.run(debug=True)
