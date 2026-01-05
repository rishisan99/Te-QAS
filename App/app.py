import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering

app = Flask(__name__)

# Model configuration
MODEL_PATH = "./Models/final_xlmr_2.0_tel_3"  # Update with your model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

# Load model and tokenizer
@app.before_first_request
def load_model():
    global model, tokenizer
    print("Loading model and tokenizer...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
    model = XLMRobertaForQuestionAnswering.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model and tokenizer loaded successfully!")

def preprocess(question, context, max_length=512):
    """
    Preprocess the question and context for the model using the same approach as training.
    """
    # Encode the input using the tokenizer
    encoding = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move tensors to the correct device
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "context": context
    }

def get_answer(encoded_inputs):
    """
    Get the answer from the model prediction.
    """
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs["attention_mask"]
        )
    
    # Get start and end logits
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()
    
    # Find the most likely answer span
    offset_mapping = encoded_inputs["offset_mapping"]
    context = encoded_inputs["context"]
    
    # Filter out CLS, SEP, or padding token positions
    attention_mask = encoded_inputs["attention_mask"][0].cpu().numpy()
    filtered_start_logits = [l if m == 1 else -float('inf') for l, m in zip(start_logits, attention_mask)]
    filtered_end_logits = [l if m == 1 else -float('inf') for l, m in zip(end_logits, attention_mask)]
    
    # Get the start and end positions with highest scores
    start_idx = filtered_start_logits.index(max(filtered_start_logits))
    end_idx = filtered_end_logits.index(max(filtered_end_logits))
    
    # Ensure valid span (end comes after start)
    if end_idx < start_idx:
        end_idx = start_idx
    
    # Get answer text
    # Check if it's a "no answer" prediction
    if start_idx == 0 and end_idx == 0:  # If pointing to CLS token
        return "No answer found in the given context."
    
    # Get character-level offsets
    start_char = offset_mapping[start_idx][0]
    end_char = offset_mapping[end_idx][1]
    
    # Extract answer from context
    if start_char and end_char:
        answer = context[start_char:end_char]
    else:
        answer = "No answer found in the given context."
    
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from request
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        
        if not question or not context:
            return jsonify({
                'error': 'Both question and context are required.'
            })
        
        # Preprocess input
        encoded_inputs = preprocess(question, context, MAX_LENGTH)
        
        # Get answer
        answer = get_answer(encoded_inputs)
        
        return jsonify({
            'question': question,
            'answer': answer
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 