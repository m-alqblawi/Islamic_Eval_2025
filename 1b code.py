import time

import joblib
from langchain_openai import ChatOpenAI
from pyarabic.araby import strip_tashkeel, strip_tatweel
from pyarabic.normalize import normalize_lamalef

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key="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"  # Replace with your actual OpenAI API key
)

import json
from tqdm import tqdm
import os

import json
from tqdm import tqdm
import os

import json

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)




def clean_text(text: str) -> str:
    if not text:
        return ""

    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_lamalef(text)
    return text.strip()


with open("submission_top20_matches_no_diacritics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

system_prompt = SystemMessagePromptTemplate.from_template(
    """
You are a highly knowledgeable expert in Quranic and Hadith text verification.

You will be given two texts:

- "query_text": This text may contain errors, partial phrases, or slight variations and is NOT guaranteed to be an exact excerpt from the Quran or Hadith.
- "candidate_text": This is a literal, exact excerpt taken from either the Quran or Hadith, free from errors.

Your task:

1. Ignore all Arabic diacritics (tashkeel) in both texts during comparison.
2. For Quranic verses ("ayah_text"), require strict literal substring matching ignoring diacritics and spacing.
3. For Hadith texts ("hadithTxt"), allow slight leniency in wording or conversational phrasing—small paraphrases or reordering are acceptable—but the core meaning and most of the key phrases should be clearly present.
4. Respond ONLY with a single word:
   - "True" if the candidate text validly matches the query according to the above criteria.
   - "False" otherwise.

Examples:

Quran Example 1:  
query_text: "يسرنا القرآن للذكر"  
candidate_text: "ولقد يسرنا القرآن للذكر فهل من مدكر"  
Answer: True  
Explanation: Literal substring present ignoring diacritics.

Quran Example 2:  
query_text: "لقد أرسلنا من قبلك رسلا وآتيناهم أيات ودافعنا عنهم الذين كفروا وكنا لهم عضد"  
candidate_text: "ولقد أرسلنا من قبلك في شيع الأولين"  
Answer: False  
Explanation: No exact substring match.

Hadith Example 1:  
query_text: "ما يصيب المؤمن من شوكة فما فوقها إلا رفعه الله بها درجة، أو حط عنه بها خطيئة"  
candidate_text: "حدثنا محمد بن عبد الله بن نمير قال رسول الله صلى الله عليه وسلم لا تصيب المؤمن شوكة فما فوقها إلا قص الله بها من خطيئته."  
Answer: True  
Explanation: Despite slight wording differences, core meaning and key phrases are clearly present with acceptable phrasing variations.

Hadith Example 2:  
query_text: "إذا مات المؤمن انتقل إلى الجنة مباشرة"  
candidate_text: "عن النبي صلى الله عليه وسلم قال: المؤمن إذا قبض توضع روحه في تاج من نور ينير ما بين المشرق والمغرب."  
Answer: False  
Explanation: Candidate text does not contain the key content or meaning of the query.

Now evaluate:

query_text: {query}  
candidate_text: {text}  
Answer:

"""
)


def merge_ayas_from_retrieval(verses_data):
    """
    Groups and merges consecutive verses from the same surah.

    Args:
        verses_data: List of dictionaries containing verse information

    Returns:
        List of merged verse groups, where consecutive verses are combined
    """
    if not verses_data:
        return []

    # Parse verse_id and add surah and ayah numbers for sorting
    parsed_verses = []
    for verse in verses_data:
        verse_id = verse['verse_id']
        surah_num, ayah_num = map(int, verse_id.split(':'))

        parsed_verse = verse.copy()
        parsed_verse['surah_num'] = surah_num
        parsed_verse['ayah_num'] = ayah_num
        parsed_verses.append(parsed_verse)

    # Sort by surah number, then by ayah number
    parsed_verses.sort(key=lambda x: (x['surah_num'], x['ayah_num']))

    merged_groups = []
    current_group = [parsed_verses[0]]

    for i in range(1, len(parsed_verses)):
        current_verse = parsed_verses[i]
        last_verse_in_group = current_group[-1]

        # Check if current verse is consecutive to the last verse in current group
        same_surah = current_verse['surah_num'] == last_verse_in_group['surah_num']
        consecutive_ayah = current_verse['ayah_num'] == last_verse_in_group['ayah_num'] + 1

        if same_surah and consecutive_ayah:
            # Add to current group
            current_group.append(current_verse)
        else:
            # Start new group
            merged_groups.append(create_merged_entry(current_group))
            current_group = [current_verse]

    # Add the last group
    merged_groups.append(create_merged_entry(current_group))

    return merged_groups


def create_merged_entry(group):
    """
    Creates a merged entry from a group of consecutive verses.

    Args:
        group: List of consecutive verses from same surah

    Returns:
        Dictionary representing the merged group
    """
    if len(group) == 1:
        # Single verse, return as is (remove helper fields)
        verse = group[0].copy()
        verse.pop('surah_num', None)
        verse.pop('ayah_num', None)
        return verse

    # Multiple verses - merge them
    first_verse = group[0]
    last_verse = group[-1]

    # Create range for verse_id (e.g., "2:155-157")
    if first_verse['ayah_num'] == last_verse['ayah_num']:
        merged_verse_id = first_verse['verse_id']
    else:
        merged_verse_id = f"{first_verse['surah_num']}:{first_verse['ayah_num']}-{last_verse['ayah_num']}"

    # Combine ayah texts
    combined_text = ' '.join([verse['ayah_text'] for verse in group])

    # Use the highest similarity score in the group
    max_similarity = max([verse['similarity_score'] for verse in group])

    merged_entry = {
        'verse_id': merged_verse_id,
        'similarity_score': max_similarity,
        'surah_name': first_verse['surah_name'],
        'ayah_text': combined_text,
        'merged_count': len(group),
        'original_verses': [verse['verse_id'] for verse in group]
    }

    return merged_entry


# Load existing results if they exist
existing_results = {}
output_file = "final.jbl"
if os.path.exists(output_file):
    existing_data = joblib.load(output_file)
    for item in existing_data:
        composite_key = f"{item['sequence_id']}"
        existing_results[composite_key] = item

human_prompt = HumanMessagePromptTemplate.from_template(
    "query_text: {query}\nCandidate Text: {text}\nAnswer:"
)
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

output = []
for item in tqdm(data, desc="Processing queries"):
    sequence_id = item["sequence_id"]
    query_id = item["question_id"]
    query_text = item["query_text"]
    span_type = item["span_type"]
    match_list = item["top_20_match_details"]
    if span_type == "Ayah":
        match_list_new = merge_ayas_from_retrieval(match_list)
    else:
        match_list_new = match_list
    # Create composite key for unique identification
    composite_key = f"{sequence_id}"
    is_merged = len(match_list) != len(match_list_new)
    # Check if this combination already exists in results
    if composite_key in existing_results and not is_merged:
        print(f"Skipping query_id {sequence_id} - already processed")
        output.append(existing_results[composite_key])
        continue
    match_list = match_list_new
    if span_type == "Ayah":
        text_key = "ayah_text"
    elif span_type == "Hadith":
        text_key = "hadithTxt"
    else:
        raise ValueError(f"Unknown span_type: {span_type}")

    processed_matches = []
    for match in match_list:
        candidate_text = match[text_key]
        messages = prompt.format_prompt(
            query=query_text, text=candidate_text
        ).to_messages()
        found_in_existing = []
        if composite_key in existing_results:
            found_in_existing = [ent for ent in existing_results[composite_key]['matches']
                                 if ent["ayah_text"] == candidate_text and "detection" in ent]

        if found_in_existing:
            match_with_detection = found_in_existing[0]
            processed_matches.append(match_with_detection)
            print("reusing previous prompt")
        else:
            response = llm.invoke(messages)
            answer = response.content.strip().lower()
            detection = True if answer == "true" else False

            match_with_detection = match.copy()
            match_with_detection["detection"] = detection
            processed_matches.append(match_with_detection)
            if detection:
                print(f"matched!! {candidate_text} \n\n {query_text}\n\n{'=' * 50}")
                break

    output.append({
        "id": query_id,
        "query": query_text,
        "sequence_id": sequence_id,
        "span_type": span_type,
        "matches": processed_matches,
    })

    # Save result
    sved = time.time()
    joblib.dump(output, f"{sved}.jbl", compress=5)
    print(f"✅ Detection complete. Saved to {sved}.jbl")
joblib.dump(output, f"final.jbl", compress=5)
