import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("humartin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humartin/chatgpt_paraphraser_on_T5_base").to(device)

API_URL = "https://api-inference.huggingface.co/models/humarin/chatgpt_paraphraser_on_T5_base"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def paraphrase(question, **kwargs):
    payload = {"inputs": f"paraphrase: {question}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_text(input_text, places_and_descs):
    generated_texts = []
    for place, desc in places_and_descs:
        paraphrased_descs = paraphrase(desc)
        if not paraphrased_descs:
            print(f"No paraphrases generated for: {desc}")
            continue
        for paraphrased_desc in paraphrased_descs:
            # Append two more sentences to each paraphrased description
            paraphrased_desc += " " + paraphrased_desc.split('.')[0] + "."
            paraphrased_desc += " " + paraphrased_desc.split('.')[1] + "."
            generated_text = f"{input_text} Welcome to {place}. {paraphrased_desc}"
            generated_texts.append(generated_text)
    return generated_texts

input_text = "Are you ready to explore some of Istanbul's iconic landmarks? Let's dive in!"

places_and_descs = [
    ("The Blue Mosque", "Welcome to The Blue Mosque. Located in the center of Istanbul, the Blue Mosque is an impressive piece of architecture that boasts six grand minarets and bright blue tiles. Inside, visitors can marvel at the intricate calligraphy, stained glass, and ceramic tiles created over the centuries, all wrapped up in a beautiful fabric wall."),
    ("Topkapi Palace", "Welcome to Topkapi Palace. Stepping into the past: A magnificent edifice on the Seraglio Point overlooking the Golden Horn, Topkapi Palace offers visitors an unparalleled view of the once-beautiful city and its rich history."),
    ("Galata Tower", "Welcome to Galata Tower. Istanbul's most renowned landmark, Galata Tower, is the ultimate destination for those seeking a slice of history and architectural wonders. Soaring over the city skyline, this tower offers breathtaking views from its medieval walls, including stunning panoramas of the Bosphorus strait, as well as captivating historical significance."),
    ("Contemporary Art Istanbul Modern", "Welcome to Contemporary Art Istanbul Modern. Located on the picturesque shores of Istanbul, Contemporary Art Istanbul Modern offers exciting opportunities for creative expression.")
]

start_time = time.time()

generated_texts = generate_text(input_text, places_and_descs)
if generated_texts:
    combined_text = "\n\n".join(generated_texts)
    print(combined_text)
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds.")
else:
    print("No text generated.")
