import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(input_text, places_and_descs):
    generated_texts = []
    for place, desc in places_and_descs:
        # Create prompt by combining input text, place, and description
        prompt = f"{input_text} Welcome to {place}. {desc}"
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate text with GPT-2
        output = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            repetition_penalty=1.2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        # Decode the generated output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Filter out unwanted parts
        generated_text = generated_text.split(f"Welcome to {place}.")[1].strip()
        
        # Remove any links and unwanted additional text
        generated_text = remove_links_and_unwanted_text(generated_text)
        
        # Filter out nonsensical sentences
        generated_text = filter_nonsense_sentences(generated_text)
        
        # Add a period at the end of each sentence
        generated_text = add_period_to_sentences(generated_text)
        
        # Add the generated text to the list
        generated_texts.append(generated_text)
    
    # Combine generated texts for all places into one with an empty line between each text
    combined_text = "\n\n".join(generated_texts)
    combined_text += "."  # Add a period to complete the last sentence
    return combined_text

# Function to remove links and unwanted additional text from text
def remove_links_and_unwanted_text(text):
    # Define unwanted patterns
    unwanted_patterns = ["<!--", "<div", "Advertisement", "All", "photos", "©", "www"]
    
    # Remove links and unwanted additional text
    cleaned_text = text
    for pattern in unwanted_patterns:
        cleaned_text = cleaned_text.split(pattern)[0]
    return cleaned_text.strip()

# Function to filter out nonsensical sentences
def filter_nonsense_sentences(text):
    # Split the text into sentences
    sentences = text.split(".")
    # Filter out sentences that are too short or nonsensical
    filtered_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
    # Join the filtered sentences back into a single text
    filtered_text = ". ".join(filtered_sentences)
    return filtered_text

# Function to add a period at the end of each sentence
def add_period_to_sentences(text):
    # Split the text into sentences
    sentences = text.split(".")
    # Add a period to the end of each sentence if it doesn't already have one
    sentences_with_periods = [sentence.strip() + "." if sentence.strip() else "" for sentence in sentences]
    # Join the sentences back into a single text
    text_with_periods = " ".join(sentences_with_periods)
    return text_with_periods

# Example usage
input_text = "Are you ready to explore some of Istanbul's iconic landmarks? Let's dive in!"

# List to store place names and descriptions
places_and_descs = [
    ("The Blue Mosque", "Welcome to the Blue Mosque, an architectural gem nestled in the heart of Istanbul. With its six majestic minarets and striking blue tiles, this masterpiece of Ottoman design beckons visitors to admire its beauty. Step inside to discover a symphony of intricate calligraphy, dazzling stained glass, and over 20,000 handcrafted ceramic tiles, all bathed in a serene atmosphere. Whether you're drawn by its historical significance or simply seeking inspiration, the Blue Mosque promises an unforgettable experience that will leave you captivated by its timeless splendor."),
    ("Topkapi Palace", "Welcome to Topkapi Palace, a magnificent testament to the grandeur of the Ottoman Empire. Perched on the Seraglio Point overlooking the shimmering Golden Horn, this opulent palace invites you to step back in time and immerse yourself in centuries of history. Explore its lush gardens, opulent courtyards, and exquisite architecture as you uncover the secrets of its royal residents. From the breathtaking views to the intricate details of its interior, Topkapi Palace offers a glimpse into a bygone era of splendor and majesty."),
    ("Galata Tower", "Welcome to Galata Tower, an iconic symbol of Istanbul's rich history and architectural legacy. Rising majestically above the city skyline, this medieval tower offers panoramic views that stretch as far as the eye can see. From its ancient walls, you can gaze out upon the bustling streets below and marvel at the beauty of the Bosphorus strait. Whether you're drawn by its historical significance or simply seeking a breathtaking vista, Galata Tower promises an unforgettable experience that will leave you in awe of Istanbul's timeless charm."),
    ("Contemporary Art Istanbul Modern", "Welcome to Contemporary Art Istanbul Modern, a vibrant hub for cutting-edge creativity in the heart of Istanbul. Nestled along the picturesque shores")]

# Start the timer
start_time = time.time()

# Generate text
generated_text = generate_text(input_text, places_and_descs)
print(generated_text)

# End the timer
end_time = time.time()

# Calculate and print the runtime
print(f"Total runtime: {end_time - start_time} seconds.")
