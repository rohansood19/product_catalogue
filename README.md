# Setup Environment

```bash
git clone git@github.com:rohansood19/product_catalogue.git
cd generative-product-catalogue-solution
conda create --name myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

# Text Deduplication and Description Generator

This repository contains a pipeline for loading, processing, deduplicating, and evaluating text data, and finally generating Master Catalogue. In addition for data preprocessing I also have used generative AI to fill missing descriptions.

##  Project Structure

```
.
├── Models.py               # Model architecture definitions
├── DataLoaders.py          # Data loading utilities
├── DataProcessor.py        # Deduplication and preprocessing logic
├── Evaluator.py            # Model evaluation functions
├── GenerateDescription.py  # Description generation logic
├── settings.py             # Configuration settings
├── Catalogue.py            # Generates the deduplicated catalogue
               
```


### 3. Run Test Script

```bash
python test.py
```

This will load sample data, process it, apply deduplication, generate descriptions, and output evaluation metrics.




