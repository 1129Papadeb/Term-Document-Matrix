import wikipediaapi
import os

# Create output folder
output_folder = "wiki_articles"
os.makedirs(output_folder, exist_ok=True)

# Initialize Wikipedia API with a proper user agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyApp/1.0 (https://example.com)' 
)

# List of Wikipedia article titles
titles = ["Artificial_intelligence", "Climate_change", "Quantum_computing", "Photosynthesis", "Blockchain"] 

# Loop through each title and download the article
for title in titles:
    page = wiki.page(title)
    if page.exists():
        # Clean filename
        filename = f"{title}.txt".replace(" ", "_")
        filepath = os.path.join(output_folder, filename)

        # Write article content to a text file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(page.text)

        print(f"Saved: {filepath}")
    else:
        print(f"Page not found: {title}")
