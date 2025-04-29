# %%capture
# !git clone https://github.com/AI4Bharat/IndicTrans2.git

# %%capture
# %cd IndicTrans2/huggingface_interface

# %%capture
# !python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
# !python3 -c "import nltk; nltk.download('punkt')"
# !python3 -m pip install bitsandbytes scipy accelerate datasets
# !python3 -m pip install sentencepiece

# !git clone https://github.com/VarunGumma/IndicTransToolkit.git
# %cd IndicTransToolkit
# !python3 -m pip install --editable ./
# %cd ..

import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from tqdm import tqdm
import os

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model_and_tokenizer(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True
    ).to(DEVICE)

    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i: i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def translate_squad_dataset_in_parts(squad_data, model, tokenizer, ip, output_file="translated_squad.json"):
    src_lang, tgt_lang = "eng_Latn", "tel_Telu"

    # Load the existing translated dataset if available
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            translated_data = json.load(f)
        last_article_idx, last_paragraph_idx, last_qa_idx = find_last_translated_position(translated_data)
        last_question_id = get_last_question_id(translated_data, last_article_idx, last_paragraph_idx, last_qa_idx)
        print(f"Resuming translation from Article: {last_article_idx}, Paragraph: {last_paragraph_idx}, QA: {last_qa_idx}")
        print(f"Last translated question ID: {last_question_id}")
    else:
        translated_data = {"version": squad_data["version"], "data": []}
        last_article_idx, last_paragraph_idx, last_qa_idx = -1, -1, -1
        last_question_id = None

    total_articles = len(squad_data["data"])
    total_contexts = sum(len(article["paragraphs"]) for article in squad_data["data"])

    translated_articles = len(translated_data["data"])
    translated_contexts = sum(len(article["paragraphs"]) for article in translated_data["data"])

    article_progress = tqdm(total=total_articles, initial=translated_articles, desc="Translating titles/articles")
    context_progress = tqdm(total=total_contexts, initial=translated_contexts, desc="Translating contexts")

    for article_idx, article in enumerate(squad_data["data"]):
        if article_idx < last_article_idx:
            continue

        if article_idx == last_article_idx:
            translated_article = translated_data["data"][article_idx]
        else:
            translated_article = {"title": "", "paragraphs": []}
            translated_article['title'] = batch_translate([article['title']], src_lang, tgt_lang, model, tokenizer, ip)[0]

        for paragraph_idx, paragraph in enumerate(article['paragraphs']):
            if article_idx == last_article_idx and paragraph_idx < last_paragraph_idx:
                continue

            if article_idx == last_article_idx and paragraph_idx == last_paragraph_idx:
                translated_paragraph = translated_article["paragraphs"][paragraph_idx]
            else:
                translated_paragraph = {
                    "context": "",
                    "qas": []
                }
                translated_paragraph['context'] = batch_translate([paragraph['context']], src_lang, tgt_lang, model, tokenizer, ip)[0]
                print(f"Translated Context:\n{translated_paragraph['context']}\n")

            for qa_idx, qa in enumerate(paragraph['qas']):
                if article_idx == last_article_idx and paragraph_idx == last_paragraph_idx and qa_idx <= last_qa_idx:
                    continue

                translated_qa = {
                    "question": "",
                    "id": qa['id'],
                    "answers": [],
                    "is_impossible": qa['is_impossible']
                }

                translated_qa['question'] = batch_translate([qa['question']], src_lang, tgt_lang, model, tokenizer, ip)[0]

                if not qa['is_impossible']:
                    for answer in qa['answers']:
                        translated_answer = {
                            "text": "",
                            "answer_start": answer['answer_start']
                        }
                        translated_answer['text'] = batch_translate([answer['text']], src_lang, tgt_lang, model, tokenizer, ip)[0]
                        translated_qa['answers'].append(translated_answer)
                else:
                    translated_qa['answers'] = []
                    if 'plausible_answers' in qa:
                        translated_qa['plausible_answers'] = []
                        for plausible_answer in qa['plausible_answers']:
                            translated_plausible_answer = {
                                "text": "",
                                "answer_start": plausible_answer['answer_start']
                            }
                            translated_plausible_answer['text'] = batch_translate([plausible_answer['text']], src_lang, tgt_lang, model, tokenizer, ip)[0]
                            translated_qa['plausible_answers'].append(translated_plausible_answer)

                print(f"Translated QA Pair:\nQuestion: {translated_qa['question']}")
                if translated_qa['answers']:
                    print(f"Answer: {translated_qa['answers'][0]['text']}\n")
                elif 'plausible_answers' in translated_qa and translated_qa['plausible_answers']:
                    print(f"Plausible Answer: {translated_qa['plausible_answers'][0]['text']} (Impossible question)\n")
                else:
                    print("No answer or plausible answer provided (Impossible question)\n")

                translated_paragraph['qas'].append(translated_qa)

            if paragraph_idx >= len(translated_article["paragraphs"]):
                translated_article["paragraphs"].append(translated_paragraph)
            context_progress.update(1)

            # Save after each context
            if article_idx >= len(translated_data["data"]):
                translated_data["data"].append(translated_article)

            with open(output_file, "w") as outfile:
                json.dump(translated_data, outfile, ensure_ascii=False, indent=4)
            print(f"Progress saved after translating context with Q&A ID: {paragraph['qas'][-1]['id']}")

        article_progress.update(1)

    # Save the final dataset
    with open(output_file, "w") as outfile:
        json.dump(translated_data, outfile, ensure_ascii=False, indent=4)
    print("Final dataset saved.")

    article_progress.close()
    context_progress.close()

def find_last_translated_position(translated_data):
    """ Find the last translated position (article, paragraph, QA) """
    if len(translated_data["data"]) > 0:
        last_article_idx = len(translated_data["data"]) - 1
        last_article = translated_data["data"][last_article_idx]

        if len(last_article["paragraphs"]) > 0:
            last_paragraph_idx = len(last_article["paragraphs"]) - 1
            last_paragraph = last_article["paragraphs"][last_paragraph_idx]

            if len(last_paragraph["qas"]) > 0:
                last_qa_idx = len(last_paragraph["qas"]) - 1
                return last_article_idx, last_paragraph_idx, last_qa_idx

            return last_article_idx, last_paragraph_idx, -1

        return last_article_idx, -1, -1

    return -1, -1, -1

def get_last_question_id(translated_data, article_idx, paragraph_idx, qa_idx):
    """ Get the ID of the last translated question """
    if article_idx >= 0 and paragraph_idx >= 0 and qa_idx >= 0:
        return translated_data["data"][article_idx]["paragraphs"][paragraph_idx]["qas"][qa_idx]["id"]
    return None

def find_last_q_id(translated_data):
    """ Find the last Q&A ID from the translated file to resume from """
    if len(translated_data["data"]) > 0:
        last_article = translated_data["data"][-1]
        if len(last_article["paragraphs"]) > 0:
            last_paragraph = last_article["paragraphs"][-1]
            if len(last_paragraph["qas"]) > 0:
                return last_paragraph["qas"][-1]["id"]
    return None

### Progress Track


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

def count_items(data):
    total_articles = len(data['data'])
    total_paragraphs = sum(len(article['paragraphs']) for article in data['data'])
    total_qa_pairs = sum(
        len(qa)
        for article in data['data']
        for paragraph in article['paragraphs']
        for qa in paragraph['qas']
    )
    return total_articles, total_paragraphs, total_qa_pairs

def check_translation_progress(translated_file_path, original_file_path):
    translated_data = load_json_file(translated_file_path)
    original_data = load_json_file(original_file_path)

    if translated_data is None or original_data is None:
        return

    translated_articles, translated_paragraphs, translated_qa_pairs = count_items(translated_data)
    original_articles, original_paragraphs, original_qa_pairs = count_items(original_data)

    print(f"Translation Progress:")
    print(f"Articles: {translated_articles}/{original_articles} ({translated_articles/original_articles:.2%})")
    print(f"Paragraphs: {translated_paragraphs}/{original_paragraphs} ({translated_paragraphs/original_paragraphs:.2%})")
    print(f"Q&A pairs: {translated_qa_pairs}/{original_qa_pairs} ({translated_qa_pairs/original_qa_pairs:.2%})")

    if translated_articles > 0:
        last_article = translated_data['data'][-1]
        last_article_title = last_article['title']
        print(f"\nLast translated article title: {last_article_title}")

        if len(last_article['paragraphs']) > 0:
            last_paragraph = last_article['paragraphs'][-1]
            last_context_preview = last_paragraph['context'][:100] + "..." if len(last_paragraph['context']) > 100 else last_paragraph['context']
            print(f"Last translated context (preview): {last_context_preview}")

            if len(last_paragraph['qas']) > 0:
                last_qa = last_paragraph['qas'][-1]
                print(f"Last translated question: {last_qa['question']}")
                if last_qa['answers']:
                    print(f"Last translated answer: {last_qa['answers'][0]['text']}")

# Load your dataset
with open("train-v2.0.json", "r") as f: #dev-v2.json
    squad_data = json.load(f)

# Initialize model, tokenizer, and processor
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir)
ip = IndicProcessor(inference=True)

# Translate the SQuAD dataset
translate_squad_dataset_in_parts(squad_data, en_indic_model, en_indic_tokenizer, ip, output_file="translated_squad_large.json")

# Clean up resources
del en_indic_tokenizer, en_indic_model