from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Önceden eğitilmiş GPT-2 modelini yükleyin
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# attention mask ve pad token id setleme
model.config.pad_token_id = tokenizer.eos_token_id

def generate_text(place, description):
    input_text = f"{place}. {description}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Metin üretmek için modeli kullanın
    output = model.generate(input_ids, 
                            max_length=200, 
                            temperature=0.7,  # Daha yüksek bir sıcaklık kullanarak daha çeşitli metinler elde edebiliriz
                            do_sample=True,
                            no_repeat_ngram_size=2,
                            top_k=50,  # Top-K sampling kullanarak daha iyi sonuçlar alabiliriz
                            top_p=0.95,  # Nucleus sampling kullanarak daha iyi sonuçlar alabiliriz
                            pad_token_id=tokenizer.eos_token_id,
                            early_stopping=True,
                            num_beams=1)  # Beam search kullanarak daha doğru sonuçlar elde edebiliriz
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

places_and_descs = [
    ("The Blue Mosque", "Located in the center of Istanbul, the Blue Mosque is an impressive piece of architecture that boasts six grand minarets and bright blue tiles. Inside, visitors can marvel at the intricate calligraphy, stained glass, and ceramic tiles created over the centuries, all wrapped up in a beautiful fabric wall."),
    ("Topkapi Palace", "Stepping into the past: A magnificent edifice on the Seraglio Point overlooking the Golden Horn, Topkapi Palace offers visitors an unparalleled view of the once-beautiful city and its rich history."),
    ("Galata Tower", "Istanbul's most renowned landmark, Galata Tower, is the ultimate destination for those seeking a slice of history and architectural wonders. Soaring over the city skyline, this tower offers breathtaking views from its medieval walls, including stunning panoramas of the Bosphorus strait, as well as captivating historical significance."),
    ("Contemporary Art Istanbul Modern", "Located on the picturesque shores of Istanbul, Contemporary Art Istanbul Modern offers exciting opportunities for creative expression."),
    ("Hagia Sophia", "Once a church, later a mosque, and now a museum, Hagia Sophia is a testament to Istanbul's rich history and architectural marvels. Its massive dome, stunning mosaics, and towering minarets draw visitors from around the world."),
    ("Bosphorus Cruise", "Embark on a journey through the heart of Istanbul with a Bosphorus Cruise. Admire the city's skyline dotted with historic landmarks, picturesque villages, and majestic bridges as you sail between Europe and Asia."),
    ("Grand Bazaar", "Experience the vibrant atmosphere of one of the world's oldest and largest covered markets, the Grand Bazaar. Lose yourself in a maze of narrow streets filled with shops selling everything from carpets and spices to jewelry and ceramics."),
    ("Dolmabahçe Palace", "A symbol of the Ottoman Empire's opulence, Dolmabahçe Palace impresses with its grandeur and European-inspired architecture. Explore its luxurious interiors, lush gardens, and stunning waterfront location on the shores of the Bosphorus."),
    ("Süleymaniye Mosque", "Dominating Istanbul's skyline, Süleymaniye Mosque is an architectural masterpiece built by the great Ottoman architect Sinan. Its massive dome, elegant minarets, and serene courtyards make it a must-visit for architecture enthusiasts.")
]

# Tüm programın çalışma süresini ölçmek için başlangıç zamanı
start_time = time.time()

# Metin üretimini başlatın
for place, desc in places_and_descs:
    generated_text = generate_text(place, desc)
    print(generated_text.strip())  # Boşlukları temizle

# Tüm programın çalışma süresini ölçmek için bitiş zamanı
end_time = time.time()

# Tüm programın çalışma süresini hesapla
total_duration = end_time - start_time
print(f"Total duration: {total_duration} seconds")
