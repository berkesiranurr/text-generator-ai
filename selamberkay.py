from textgenrnn import textgenrnn

# Veriler
places_and_descs = [
    ("The Blue Mosque", "Located in the center of Istanbul, the Blue Mosque is an impressive piece of architecture that boasts six grand minarets and bright blue tiles. Inside, visitors can marvel at the intricate calligraphy, stained glass, and ceramic tiles created over the centuries, all wrapped up in a beautiful fabric wall."),
    ("Topkapi Palace", "Stepping into the past: A magnificent edifice on the Seraglio Point overlooking the Golden Horn, Topkapi Palace offers visitors an unparalleled view of the once-beautiful city and its rich history."),
    ("Galata Tower", "Istanbul's most renowned landmark, Galata Tower, is the ultimate destination for those seeking a slice of history and architectural wonders. Soaring over the city skyline, this tower offers breathtaking views from its medieval walls, including stunning panoramas of the Bosphorus strait, as well as captivating historical significance."),
    ("Contemporary Art Istanbul Modern", "Located on the picturesque shores of Istanbul, Contemporary Art Istanbul Modern offers exciting opportunities for creative expression.")
]

# Metin üreteci modeli oluşturma ve eğitme
textgen = textgenrnn.TextgenRnn()
textgen.train_on_texts([desc for _, desc in places_and_descs], num_epochs=5)

# Her mekan için yeni bir açıklama oluşturma
for place, _ in places_and_descs:
    new_description = textgen.generate(return_as_list=True, max_gen_length=150)[0]
    print(f"{place}: {new_description}\n")
