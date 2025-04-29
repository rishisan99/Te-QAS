import json
import os
from tqdm import tqdm
from rapidfuzz import fuzz
import re

class AnswerSpanFinder:
    def __init__(self, language_config=None):
        """
        Initialize with language-specific configurations
        """
        # Default configuration - can be overridden
        self.config = {
            'connectors': ['మరియు', 'మరి', 'తో', 'లో'],  # Add more connectors
            'suffixes': [
                'లో', 'లోని', 'కి', 'తో', 'ను', 'లు', 'ని', 'కు', 'డు', 
                'రాలు', 'డిని', 'నికి', 'లను', 'లతో', 'గారు', 'వు',
                'లోపల', 'బయట', 'వైపు', 'వెంట', 'మీద', 'క్రింద'
            ],
            'variations': {
                'చైన': ['చైనా', 'చైనీస్'],
                'అమెరిక': ['అమెరికా', 'అమెరికన్'],
                'రష్య': ['రష్యా', 'రష్యన్'],
                'జపాన్': ['జపాన్', 'జపనీస్'],
                'భారత': ['భారత్', 'భారతీయ'],
                'బ్రిటన్': ['బ్రిటన్', 'బ్రిటిష్'],
                # Add more variations as needed
            },
            'semantic_groups': {
                # Approximate/About
                'సుమారు': ['సుమారు', 'దాదాపు', 'దరిదాపు', 'ఇంచుమించు', 'అంచనా', 'గణన'],
                
                # Many/More/Much
                'చాలా': ['చాలా', 'అనేక', 'పెక్కు', 'ఎక్కువ', 'అధిక', 'మరింత', 'మరిన్ని', 'విస్తారంగా'],
                
                # All/Every
                'అన్ని': ['అన్ని', 'అందరూ', 'ప్రతి', 'సమస్త', 'సకల', 'మొత్తం'],
                
                # Time expressions
                'ఇప్పుడు': ['ఇప్పుడు', 'ప్రస్తుతం', 'నేడు', 'వర్తమానం'],
                'అప్పుడు': ['అప్పుడు', 'ఆనాడు', 'ఆరోజు', 'ఆసమయం'],
                
                # Some/Few
                'కొన్ని': ['కొన్ని', 'కొందరు', 'కొంత', 'కొద్ది'],
                
                # According to/As per
                'ప్రకారం': ['ప్రకారం', 'అనుసారం', 'ననుసరించి', 'బట్టి'],
                
                # But/However
                'కానీ': ['కానీ', 'అయితే', 'గానీ', 'కాని'],
                
                # Therefore/So
                'కాబట్టి': ['కాబట్టి', 'అందుచేత', 'దీనివల్ల', 'అందువల్ల', 'కనుక'],
                
                # Again/Once more
                'మళ్ళీ': ['మళ్ళీ', 'మరల', 'తిరిగి', 'మరలా', 'మరొక్కసారి'],
                
                # Only/Just
                'మాత్రమే': ['మాత్రమే', 'కేవలం', 'పరిమితం'],
                
                # Together/With
                'కలిసి': ['కలిసి', 'కూడా', 'సహా', 'తోపాటు', 'వెంట'],
                
                # Before/Previously
                'ముందు': ['ముందు', 'పూర్వం', 'ఇంతకుముందు', 'ఇదివరకు', 'గతంలో'],
                
                # After/Later
                'తర్వాత': ['తర్వాత', 'పిమ్మట', 'అనంతరం', 'ఆ పైన'],
                
                # Between/Among
                'మధ్య': ['మధ్య', 'నడుమ', 'మధ్యన', 'మధ్యలో'],
                
                # Completely/Fully
                'పూర్తిగా': ['పూర్తిగా', 'సంపూర్ణంగా', 'మొత్తంగా', 'సమగ్రంగా'],
                
                # Immediately/At once
                'వెంటనే': ['వెంటనే', 'తక్షణమే', 'అప్పటికప్పుడు'],
                
                # Usually/Generally
                'సాధారణంగా': ['సాధారణంగా', 'మామూలుగా', 'ఎక్కువగా', 'పలుమారు'],
                
                # Supernatural/Evil beings
                'రాక్షస': ['రాక్షసుడు', 'దెయ్యం', 'భూతం', 'పిశాచి', 'దానవుడు', 'అసురుడు', 'రాక్షసి'],
                
                # Divine/Good beings
                'దేవ': ['దేవుడు', 'దైవం', 'భగవంతుడు', 'ప్రభువు', 'స్వామి'],
            },
            'min_similarity_score': 70,
            'partial_match_threshold': 80,
            'exact_match_bonus': 20,
            'char_range': (0x0C00, 0x0C7F)  # Telugu character range
        }
        
        if language_config:
            self.config.update(language_config)

    def normalize_text(self, text):
        """Normalize text while preserving character positions"""
        return re.sub(r'\s+', ' ', text.strip())

    def get_base_words(self, text):
        """
        Get base forms of words handling variations including common Telugu equivalents
        """
        if isinstance(text, str):
            words = text.split()
        else:
            words = text

        base_words = []
        for word in words:
            base_word = word

            # Remove suffixes
            for suffix in self.config['suffixes']:
                if word.endswith(suffix):
                    base_word = word[:-len(suffix)]
                    break

            # Handle variations
            found_variation = False
            for base, variants in self.config['variations'].items():
                if word in variants or word.startswith(base):
                    base_word = base
                    found_variation = True
                    break

            # Check semantic groups
            if not found_variation:
                for base, variants in self.config['semantic_groups'].items():
                    if word in variants:
                        base_word = base
                        break
                    # Check variants for partial matches
                    for variant in variants:
                        if (variant.startswith(word) or 
                            word.startswith(variant) or 
                            self._calculate_similarity(word, variant) > 0.8):
                            base_word = base
                            break

            base_words.append(base_word)
        return base_words

    def _calculate_similarity(self, word1, word2):
        """Calculate similarity between two words"""
        return fuzz.ratio(word1.lower(), word2.lower()) / 100

    def _calculate_token_similarity(self, text1, text2):
        """Calculate token-based similarity between two texts"""
        token_sort = fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
        token_set = fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100
        return (token_sort + token_set) / 2

    def _calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        words1 = self.get_base_words(text1)
        words2 = self.get_base_words(text2)
        
        matches = sum(1 for w1 in words1
                     for w2 in words2
                     if self.are_words_equivalent(w1, w2))
        
        max_len = max(len(words1), len(words2))
        return matches / max_len if max_len > 0 else 0

    def are_words_equivalent(self, word1, word2):
        """Check if two words are equivalent in meaning"""
        word1_base = self.get_base_words([word1])[0]
        word2_base = self.get_base_words([word2])[0]
        
        if word1_base == word2_base:
            return True
            
        return self._calculate_similarity(word1, word2) > 0.8

    def is_valid_char_boundary(self, text, pos):
        """Check if position is at a valid character boundary"""
        if pos <= 0 or pos >= len(text):
            return True
        char_range = self.config['char_range']
        return (ord(text[pos]) < char_range[0] or 
                ord(text[pos-1]) < char_range[0])

    def get_exact_position(self, context, substring, approximate_pos):
        """Find exact character position with language awareness"""
        window = len(substring) * 2
        start = max(0, approximate_pos - window)
        end = min(len(context), approximate_pos + window)
        
        # Adjust to valid character boundaries
        while start > 0 and not self.is_valid_char_boundary(context, start):
            start -= 1
        while end < len(context) and not self.is_valid_char_boundary(context, end):
            end += 1
            
        search_window = context[start:end]
        
        # Try exact match first
        potential_matches = []
        for i in range(len(search_window)):
            if self.is_valid_char_boundary(search_window, i):
                if search_window[i:].startswith(substring):
                    potential_matches.append((start + i, 100))
                    
        # If no exact match, try fuzzy matching
        if not potential_matches:
            for i in range(len(search_window)):
                if self.is_valid_char_boundary(search_window, i):
                    candidate = search_window[i:i+len(substring)]
                    score = fuzz.ratio(candidate, substring)
                    if score >= self.config['partial_match_threshold']:
                        potential_matches.append((start + i, score))
        
        if potential_matches:
            return max(potential_matches, key=lambda x: x[1])[0]
        return approximate_pos

    def find_best_match(self, context_words, answer_words, context, answer):
        """Find best matching span using multiple strategies"""
        best_score = 0
        best_result = None

        # Normalize and get base words
        context_bases = self.get_base_words(context_words)
        answer_bases = self.get_base_words(answer_words)

        # Try different window sizes
        min_window = len(answer_words)
        max_window = min_window + 3

        for i in range(len(context_words)):
            for window_size in range(min_window, max_window + 1):
                if i + window_size > len(context_words):
                    continue
                    
                window_words = context_words[i:i+window_size]
                window_text = ' '.join(window_words)
                window_base = ' '.join(self.get_base_words(window_words))
                
                # Calculate similarities
                exact_score = self._calculate_similarity(window_text, answer)
                token_score = self._calculate_token_similarity(window_text, answer)
                semantic_score = self._calculate_semantic_similarity(window_text, answer)
                
                score = (exact_score * 0.4 + token_score * 0.3 + semantic_score * 0.3) * 100

                # Apply bonuses
                if self._has_complete_match(window_base, answer_bases):
                    score += self.config['exact_match_bonus']
                
                if score > best_score:
                    char_pos = len(' '.join(context_words[:i]))
                    if i > 0:
                        char_pos += 1  # Account for space
                    
                    exact_pos = self.get_exact_position(context, window_text, char_pos)
                    best_score = score
                    best_result = (exact_pos, exact_pos + len(window_text), 
                                 window_text, score)
        
        return best_result or (0, 0, "", 0)

    def _has_complete_match(self, text_base, answer_bases):
        """Check if base words of text contain all base words of the answer"""
        text_base_words = set(text_base.split())
        answer_base_words = set(answer_bases)
        return answer_base_words.issubset(text_base_words)

    def find_answer_span(self, context, answer):
        """Main method to find answer span"""
        context_norm = self.normalize_text(context)
        answer_norm = self.normalize_text(answer)
        
        context_words = context_norm.split()
        answer_words = answer_norm.split()
        
        return self.find_best_match(context_words, answer_words, 
                                  context_norm, answer_norm)

def adjust_squad_dataset_spans(squad_data, output_file="adjusted_squad.json", low_similarity_file="low_similarity_cases.json", language_config=None):
    """
    Adjust answer spans in the entire SQuAD dataset with save and resume capabilities.
    Handles partial matching and semantic variations. Now always adjusts spans,
    and logs low similarity cases as well.
    """
    finder = AnswerSpanFinder(language_config)
    min_similarity_threshold = 45  # threshold for logging low similarity

    # Load the existing adjusted dataset if available
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            adjusted_data = json.load(f)
        last_article_idx, last_paragraph_idx, last_qa_idx = find_last_processed_position(adjusted_data)
        last_question_id = get_last_question_id(adjusted_data, last_article_idx, last_paragraph_idx, last_qa_idx)
        print(f"Resuming adjustment from Article: {last_article_idx}, Paragraph: {last_paragraph_idx}, QA: {last_qa_idx}")
        print(f"Last adjusted question ID: {last_question_id}")
    else:
        adjusted_data = {"version": squad_data["version"], "data": []}
        last_article_idx, last_paragraph_idx, last_qa_idx = -1, -1, -1
        last_question_id = None

    # Load the existing low similarity cases if available
    if os.path.exists(low_similarity_file):
        with open(low_similarity_file, "r", encoding="utf-8") as f:
            low_similarity_data = json.load(f)
    else:
        low_similarity_data = {"version": squad_data["version"], "data": []}

    total_articles = len(squad_data["data"])
    total_contexts = sum(len(article["paragraphs"]) for article in squad_data["data"])
    total_qas = sum(len(paragraph['qas']) for article in squad_data["data"] for paragraph in article["paragraphs"])

    adjusted_articles = len(adjusted_data["data"])
    adjusted_contexts = sum(len(article["paragraphs"]) for article in adjusted_data["data"])
    adjusted_qas = sum(len(paragraph['qas']) for article in adjusted_data["data"] for paragraph in article["paragraphs"])

    # Initialize progress bars with correct initial values
    article_progress = tqdm(total=total_articles, initial=adjusted_articles, desc="Adjusting articles")
    context_progress = tqdm(total=total_contexts, initial=adjusted_contexts, desc="Adjusting contexts")
    qa_progress = tqdm(total=total_qas, initial=adjusted_qas, desc="Adjusting questions")

    try:
        for article_idx, article in enumerate(squad_data["data"]):
            if article_idx < last_article_idx:
                continue

            if article_idx == last_article_idx and adjusted_articles > 0:
                adjusted_article = adjusted_data["data"][article_idx]
            else:
                adjusted_article = {"title": article["title"], "paragraphs": []}

            for paragraph_idx, paragraph in enumerate(article["paragraphs"]):
                if article_idx == last_article_idx and paragraph_idx < last_paragraph_idx:
                    continue

                if article_idx == last_article_idx and paragraph_idx == last_paragraph_idx and adjusted_contexts > 0:
                    adjusted_paragraph = adjusted_article["paragraphs"][paragraph_idx]
                else:
                    adjusted_paragraph = {
                        "context": paragraph['context'],
                        "qas": []
                    }

                context = paragraph['context']

                for qa_idx, qa in enumerate(paragraph['qas']):
                    if article_idx == last_article_idx and paragraph_idx == last_paragraph_idx and qa_idx <= last_qa_idx:
                        continue

                    adjusted_qa = {
                        "question": qa['question'],
                        "id": qa['id'],
                        "answers": [],
                        "is_impossible": qa['is_impossible']
                    }

                    if not qa['is_impossible']:
                        adjusted_answers = []
                        for answer in qa['answers']:
                            original_answer = answer['text']

                            start, end, found_text, score = finder.find_answer_span(context, original_answer)
                            
                            # Always update answer spans
                            adjusted_answer = {
                                "text": context[start:end],
                                "answer_start": start
                            }

                            # If score is below threshold, log as low similarity
                            if score < min_similarity_threshold:
                                low_similarity_entry = {
                                    "context": context,
                                    "qas": [{
                                        "question": qa['question'],
                                        "id": qa['id'],
                                        "answers": [answer],
                                        "found_text": found_text,
                                        "similarity_score": score
                                    }]
                                }
                                append_low_similarity_case(low_similarity_data, article['title'], low_similarity_entry)
                            
                            adjusted_answers.append(adjusted_answer)
                        adjusted_qa['answers'] = adjusted_answers
                    else:
                        adjusted_qa['answers'] = []
                        if 'plausible_answers' in qa:
                            adjusted_plausible_answers = []
                            for answer in qa['plausible_answers']:
                                original_answer = answer['text']

                                start, end, found_text, score = finder.find_answer_span(context, original_answer)

                                # Always update plausible answers
                                adjusted_answer = {
                                    "text": context[start:end],
                                    "answer_start": start
                                }

                                # If score is below threshold, log as low similarity
                                if score < min_similarity_threshold:
                                    low_similarity_entry = {
                                        "context": context,
                                        "qas": [{
                                            "question": qa['question'],
                                            "id": qa['id'],
                                            "plausible_answers": [answer],
                                            "found_text": found_text,
                                            "similarity_score": score
                                        }]
                                    }
                                    append_low_similarity_case(low_similarity_data, article['title'], low_similarity_entry)

                                adjusted_plausible_answers.append(adjusted_answer)
                            adjusted_qa['plausible_answers'] = adjusted_plausible_answers
                        else:
                            adjusted_qa['plausible_answers'] = []

                    adjusted_paragraph['qas'].append(adjusted_qa)
                    adjusted_qas += 1
                    qa_progress.update(1)

                if paragraph_idx >= len(adjusted_article["paragraphs"]):
                    adjusted_article["paragraphs"].append(adjusted_paragraph)
                adjusted_contexts += 1
                context_progress.update(1)

                # Save after each context
                if article_idx >= len(adjusted_data["data"]):
                    adjusted_data["data"].append(adjusted_article)

                with open(output_file, "w", encoding="utf-8") as outfile:
                    json.dump(adjusted_data, outfile, ensure_ascii=False, indent=4)

                with open(low_similarity_file, "w", encoding="utf-8") as outfile:
                    json.dump(low_similarity_data, outfile, ensure_ascii=False, indent=4)

            adjusted_articles += 1
            article_progress.update(1)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Saving progress...")
        # Save the adjusted data
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(adjusted_data, outfile, ensure_ascii=False, indent=4)
        # Save the low similarity cases
        with open(low_similarity_file, "w", encoding="utf-8") as outfile:
            json.dump(low_similarity_data, outfile, ensure_ascii=False, indent=4)
        print("Progress saved.")

        # Get last processed question ID
        last_article_idx, last_paragraph_idx, last_qa_idx = find_last_processed_position(adjusted_data)
        last_question_id = get_last_question_id(adjusted_data, last_article_idx, last_paragraph_idx, last_qa_idx)
        print(f"Last adjusted question ID: {last_question_id}")

        # Close progress bars
        article_progress.close()
        context_progress.close()
        qa_progress.close()
        raise

    # Save the final dataset
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(adjusted_data, outfile, ensure_ascii=False, indent=4)

    with open(low_similarity_file, "w", encoding="utf-8") as outfile:
        json.dump(low_similarity_data, outfile, ensure_ascii=False, indent=4)

    print("Final adjusted dataset saved.")

    # Close progress bars
    article_progress.close()
    context_progress.close()
    qa_progress.close()

def get_last_question_id(adjusted_data, article_idx, paragraph_idx, qa_idx):
    """ Get the ID of the last adjusted question """
    if article_idx >= 0 and paragraph_idx >= 0 and qa_idx >= 0:
        return adjusted_data["data"][article_idx]["paragraphs"][paragraph_idx]["qas"][qa_idx]["id"]
    return None

def find_last_processed_position(adjusted_data):
    """Find the last processed article, paragraph, and QA indices."""
    if len(adjusted_data["data"]) > 0:
        last_article_idx = len(adjusted_data["data"]) - 1
        last_article = adjusted_data["data"][last_article_idx]

        if len(last_article["paragraphs"]) > 0:
            last_paragraph_idx = len(last_article["paragraphs"]) - 1
            last_paragraph = last_article["paragraphs"][last_paragraph_idx]

            if len(last_paragraph["qas"]) > 0:
                last_qa_idx = len(last_paragraph["qas"]) - 1
                return last_article_idx, last_paragraph_idx, last_qa_idx

            return last_article_idx, last_paragraph_idx, -1

        return last_article_idx, -1, -1

    return -1, -1, -1

def append_low_similarity_case(low_similarity_data, article_title, low_similarity_entry):
    """
    Append a low similarity case to the low similarity data, maintaining the SQuAD 2.0 structure.
    """
    # Check if the article already exists in low_similarity_data
    for article in low_similarity_data['data']:
        if article['title'] == article_title:
            # Article found, append paragraph
            article['paragraphs'].append(low_similarity_entry)
            return

    # Article not found, create new
    new_article = {
        "title": article_title,
        "paragraphs": [low_similarity_entry]
    }
    low_similarity_data['data'].append(new_article)

# Usage Example

# Load your translated SQuAD 2.0 dataset
with open("transformed_squad_low_sim_modified.json", "r", encoding="utf-8") as f:
    translated_squad_data = json.load(f)

# Adjust the answer spans
adjust_squad_dataset_spans(translated_squad_data, output_file="adjusted_squad_nums_low_sim.json", low_similarity_file="low_two_similarity_cases.json")