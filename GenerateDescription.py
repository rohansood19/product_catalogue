import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tqdm.pandas()

class GenerateDescription:
    tqdm.pandas()
    def generator(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    
    def generate_description(self, name, category, model, tokenizer):
        keywords = f"{name}, {category}"
        prompt = f"Generate a product description from keywords: {keywords}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        output = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def generate(self, data, model, tokenizer, save_path='generated_descriptions_data.csv'):
        data['filled_description'] = data.progress_apply(
        lambda row: row['description']
        if pd.notnull(row['description'])
        else self.generate_description(row['name'], row['category'], model, tokenizer),
        axis=1
        )
        data.to_csv(save_path, index=False)
        return data
    
