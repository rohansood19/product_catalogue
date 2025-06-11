import os
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from itertools import product
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class ModelFaiss:
    
    def combine_product_fields(self, product):
        fields = [product.get("name", ""),
                product.get("features", {}).get("filled_description") or product.get("features", {}).get("description", ""),
                product.get("features", {}).get("category", "")]
        return " ".join([f for f in fields if f])
    
    def ensure_faiss_compatible(self, arr):
        return np.ascontiguousarray(np.array(arr, dtype='float32'))

    def generate_embeddings(self, products, model):
        texts = [self.combine_product_fields(p) for p in products]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return self.ensure_faiss_compatible(embeddings)


    def deduplicate(self, products_a, products_b, model, threshold=0.8):
        print("Encoding Dataset A...")
        emb_a = self.generate_embeddings(products_a, model)
        print("Encoding Dataset B...")
        emb_b = self.generate_embeddings(products_b, model)

        print("Building FAISS Index...")
        dim = emb_b.shape[1]
        index = faiss.IndexFlatIP(dim)

        emb_b = self.ensure_faiss_compatible(emb_b)
        faiss.normalize_L2(emb_b)
        index.add(emb_b)

        emb_a = self.ensure_faiss_compatible(emb_a)
        faiss.normalize_L2(emb_a)

        print("Searching for Matches...")
        D, I = index.search(emb_a, 10)

        matches = []
        for i in range(len(D)):
            for j, score in zip(I[i], D[i]):
                if score >= threshold:
                    matches.append((i, j, float(score)))
        return matches
    
    def get_matched_products(self, products_a, products_b, model:str = 'all-MiniLM-L6-v2', threshold=0.8):
        model = SentenceTransformer(model)
        matches = self.deduplicate(products_a, products_b, model, threshold=threshold)
        return matches
    
    def viewMatches(self, matches, products_a, products_b):
        matched_products = []
        for idx_a, idx_b, score in matches:
            matched_products.append({
                "product_a_name": products_a[idx_a].get("name", ""),
                "product_b_name": products_b[idx_b].get("name", ""),
                "product_a_description": products_a[idx_a].get("features", {}).get("description", ""),
                "product_b_description": products_b[idx_b].get("features", {}).get("description", ""),
                "product_a_category": products_a[idx_a].get("features", {}).get("category", ""),
                "product_b_category": products_b[idx_b].get("features", {}).get("category", ""),
                "match_score": round(score, 2)
            })
        
        df = pd.DataFrame(matched_products)
        
        # Print matches in the requested format
        for _, row in df.iterrows():
            print(f"Match Found: {row['product_a_name']} <--> {row['product_b_name']} | Score: {row['match_score']:.2f}")
        
        return df
    
class ModelSemanticSearch:
        
    def combine_product_fields(self, product):
        fields = [product.get("name", ""),
                product.get("features", {}).get("filled_description") or product.get("features", {}).get("description", ""),
                product.get("features", {}).get("category", "")]
        return " ".join([f for f in fields if f])

    def generate_embeddings(self, products, model):
        texts = [self.combine_product_fields(p) for p in products]
        return model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def deduplicate_fast(self, products_a, products_b, model, threshold=0.8, batch_size=100):

        print("Encoding Dataset A...")
        emb_a = self.generate_embeddings(products_a, model)
        print("Encoding Dataset B...")
        emb_b = self.generate_embeddings(products_b, model)

        matches = []
        print("Computing Cosine Similarity in Batches...")

        for i in tqdm(range(0, len(emb_a), batch_size), desc="Processing Batches A"):
            end_i = min(i + batch_size, len(emb_a))
            emb_a_batch = emb_a[i:end_i]

            # Compute cosine similarity between emb_a_batch and all of emb_b
            sim_matrix = util.cos_sim(emb_a_batch, emb_b)

            # Find all matches above threshold
            matched_indices = torch.nonzero(sim_matrix > threshold)
            for idx in matched_indices:
                idx_a = i + idx[0].item()  # Offset by batch start
                idx_b = idx[1].item()
                score = sim_matrix[idx[0], idx[1]].item()
                matches.append((idx_a, idx_b, score))

        return matches
    
    def get_matched_products(self, products_a, products_b, model:str = 'all-MiniLM-L6-v2', threshold=0.8):
        model = SentenceTransformer(model)
        matches = self.deduplicate_fast(products_a, products_b, model, threshold=threshold)
        return matches
    
    def viewMatches(self, matches, products_a, products_b):
        matched_products = []
        for idx_a, idx_b, score in matches:
            matched_products.append({
                "product_a_name": products_a[idx_a].get("name", ""),
                "product_b_name": products_b[idx_b].get("name", ""),
                "product_a_description": products_a[idx_a].get("features", {}).get("description", ""),
                "product_b_description": products_b[idx_b].get("features", {}).get("description", ""),
                "product_a_category": products_a[idx_a].get("features", {}).get("category", ""),
                "product_b_category": products_b[idx_b].get("features", {}).get("category", ""),
                "match_score": round(score, 2)
            })
        
        df = pd.DataFrame(matched_products)
        
        # Print matches in the requested format
        for _, row in df.iterrows():
            print(f"Match Found: {row['product_a_name']} <--> {row['product_b_name']} | Score: {row['match_score']:.2f}")
        
        return df


class ModelTFIDF:
    def __init__(self, threshold=0.8, batch_size=100):
        self.threshold = threshold
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def combine_product_fields(self, product):
        fields = [
            product.get("name", ""),
            product.get("features", {}).get("filled_description") or 
            product.get("features", {}).get("description", ""),
            product.get("features", {}).get("category", "")
        ]
        return " ".join([str(f) for f in fields if f])

    def generate_embeddings(self, products):
        texts = [self.combine_product_fields(p) for p in products]
        return self.vectorizer.fit_transform(texts)

    def deduplicate(self, products_a, products_b):
        print("Vectorizing Dataset A...")
        emb_a = self.generate_embeddings(products_a)
        
        print("Vectorizing Dataset B...")
        emb_b = self.vectorizer.transform([self.combine_product_fields(p) for p in products_b])
        
        matches = []
        total_batches = (len(products_a) + self.batch_size - 1) // self.batch_size
        
        print("Computing TF-IDF Similarity in Batches...")
        for i in tqdm(range(0, len(products_a), self.batch_size), 
                     desc="Processing Batches", 
                     total=total_batches):
            end_i = min(i + self.batch_size, len(products_a))
            batch_a = emb_a[i:end_i]
            
            # Compute cosine similarity for the batch
            similarity_matrix = cosine_similarity(batch_a, emb_b)
            
            # Find matches above threshold
            for batch_idx, similarities in enumerate(similarity_matrix):
                doc_idx = i + batch_idx
                matches.extend([
                    (doc_idx, j, float(score))
                    for j, score in enumerate(similarities)
                    if score >= self.threshold
                ])
        
        return matches

    def get_matched_products(self, products_a, products_b, threshold=0.8):
        self.threshold = threshold
        matches = self.deduplicate(products_a, products_b)
        return matches

    def viewMatches(self, matches, products_a, products_b):
        matched_products = []
        for idx_a, idx_b, score in matches:
            matched_products.append({
                "product_a_name": products_a[idx_a].get("name", ""),
                "product_b_name": products_b[idx_b].get("name", ""),
                "product_a_description": products_a[idx_a].get("features", {}).get("description", ""),
                "product_b_description": products_b[idx_b].get("features", {}).get("description", ""),
                "product_a_category": products_a[idx_a].get("features", {}).get("category", ""),
                "product_b_category": products_b[idx_b].get("features", {}).get("category", ""),
                "match_score": round(score, 2)
            })
        
        df = pd.DataFrame(matched_products)
        
        # Print matches in the requested format
        for _, row in df.iterrows():
            print(f"Match Found: {row['product_a_name']} <--> {row['product_b_name']} | Score: {row['match_score']:.2f}")
        
        return df
    
class ModelBERT:
    def __init__(self, model_name='sentence-transformers/all-distilroberta-v1', threshold=0.8, batch_size=128):
        # CUDA optimizations for A100
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
            torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cudnn
            torch.backends.cudnn.benchmark = True         # Enable cudnn autotuner
            torch.backends.cudnn.enabled = True           # Enable cudnn
            
            # Set architecture flags for A100
            torch.cuda.set_device(0)  # Use first GPU
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  
        
        self.threshold = threshold
        self.batch_size = batch_size  # Increased for A100's larger memory
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
            # Enable mixed precision training
            self.scaler = torch.cuda.amp.GradScaler()
            # Optional: Convert to half precision for more speed
            self.model = self.model.half()
            
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("TF32 Enabled: ", torch.backends.cuda.matmul.allow_tf32)
            
    def combine_product_fields(self, product):
        fields = [product.get("name", ""),
                product.get("features", {}).get("filled_description") or product.get("features", {}).get("description", ""),
                product.get("features", {}).get("category", "")]
        return " ".join([f for f in fields if f])
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, products):
        texts = [self.combine_product_fields(p) for p in products]
        embeddings = []
        
        # Increase batch size for GPU efficiency
        with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating BERT embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,  # Limit sequence length
                    return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                outputs = self.model(**encoded)
                batch_embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
                embeddings.append(batch_embeddings.cpu())
                
                # Clear cache periodically
                if i % (self.batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        return torch.cat(embeddings, dim=0)

    def deduplicate(self, products_a, products_b):
        print("Generating BERT embeddings for Dataset A...")
        emb_a = self.generate_embeddings(products_a)
        
        print("Generating BERT embeddings for Dataset B...")
        emb_b = self.generate_embeddings(products_b)
        
        matches = []
        print("Computing Cosine Similarity in Batches...")
        
        # Reduce batch size for similarity computation
        similarity_batch_size = min(32, self.batch_size)  # Smaller batches for similarity
        
        # Use torch.amp.autocast instead of deprecated cuda.amp.autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for i in tqdm(range(0, len(emb_a), similarity_batch_size), 
                        desc="Computing similarities",
                        total=len(emb_a)//similarity_batch_size):
                
                end_i = min(i + similarity_batch_size, len(emb_a))
                batch_a = emb_a[i:end_i].to(self.device)
                
                # Process B in smaller chunks
                chunk_size = similarity_batch_size * 2
                for j in range(0, len(emb_b), chunk_size):
                    end_j = min(j + chunk_size, len(emb_b))
                    batch_b = emb_b[j:end_j].to(self.device)
                    
                    # Compute similarity more efficiently
                    similarity = torch.mm(batch_a, batch_b.t())
                    similarity = similarity / (
                        batch_a.norm(dim=1)[:, None] * batch_b.norm(dim=1)[None, :]
                    )
                    
                    # Find matches efficiently
                    matches_mask = similarity > self.threshold
                    indices = torch.nonzero(matches_mask)
                    
                    if len(indices) > 0:
                        for idx in indices:
                            idx_a = i + idx[0].item()
                            idx_b = j + idx[1].item()
                            score = similarity[idx[0], idx[1]].item()
                            matches.append((idx_a, idx_b, score))
                    
                    # Clear memory
                    del batch_b, similarity
                    torch.cuda.empty_cache()
                
                # Clear memory after processing each batch_a
                del batch_a
                torch.cuda.empty_cache()
        
        print(f"Found {len(matches)} matches")
        return matches

    def get_matched_products(self, products_a, products_b, threshold=0.8):
        """
        Get matches between two product sets using BERT embeddings with GPU acceleration
        """
        print("\nInitializing BERT matching process...")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        
        # Update threshold if provided
        self.threshold = threshold
        
        # Get matches using GPU-optimized deduplication
        matches = self.deduplicate(products_a, products_b)
        
        # Sort matches by score in descending order
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
        
        print(f"\nFound {len(sorted_matches)} matches above threshold {threshold}")
        return sorted_matches

    def viewMatches(self, matches, products_a, products_b):
        """
        View matched products with detailed information and progress tracking
        """
        print("\nProcessing matches for viewing...")
        matched_products = []
        
        for idx_a, idx_b, score in tqdm(matches, desc="Processing matches"):
            # Get product details with error handling
            try:
                product_a = products_a[idx_a]
                product_b = products_b[idx_b]
                
                matched_products.append({
                    "product_a_name": product_a.get("name", ""),
                    "product_b_name": product_b.get("name", ""),
                    "product_a_description": product_a.get("features", {}).get("description", ""),
                    "product_b_description": product_b.get("features", {}).get("description", ""),
                    "product_a_category": product_a.get("features", {}).get("category", ""),
                    "product_b_category": product_b.get("features", {}).get("category", ""),
                    "match_score": round(score, 2)
                })
            except (IndexError, KeyError) as e:
                print(f"Warning: Error processing match ({idx_a}, {idx_b}): {str(e)}")
                continue
        
        # Create DataFrame with matches
        df = pd.DataFrame(matched_products)
        
        # Print summary statistics
        print("\nMatch Summary:")
        print(f"Total matches found: {len(matches)}")
        print(f"Average match score: {df['match_score'].mean():.2f}")
        print(f"Score range: {df['match_score'].min():.2f} - {df['match_score'].max():.2f}")
        
        # Print detailed matches with formatting
        print("\nDetailed Matches:")
        for _, row in df.iterrows():
            print(f"Match Found: {row['product_a_name']} <--> {row['product_b_name']} | Score: {row['match_score']:.2f}")
            
        return df

class ModelEnsemble:
    def __init__(self, models=None, weights=None):
        self.models = models or {
            'faiss': ModelFaiss(),
            'semantic': ModelSemanticSearch(),
            'bert': ModelBERT()
        }
        self.weights = weights or {
            'faiss': 0.45,
            'semantic': 0.50,
            'bert': 0.1
        }
    
    def get_matched_products(self, products_a, products_b, threshold=0.8):
        all_matches = {}
        final_matches = []
        
        # Get matches from each model
        print("Getting matches from individual models...")
        for model_name, model in self.models.items():
            print(f"\nProcessing {model_name.upper()} model...")
            matches = model.get_matched_products(products_a, products_b, threshold=threshold)
            all_matches[model_name] = {(m[0], m[1]): m[2] for m in matches}
        
        # Combine matches using weighted voting
        print("\nCombining results...")
        unique_pairs = set().union(*[set(m.keys()) for m in all_matches.values()])
        
        for pair in tqdm(unique_pairs, desc="Combining matches"):
            weighted_score = 0
            total_weight = 0
            
            for model_name, matches in all_matches.items():
                if pair in matches:
                    weight = self.weights[model_name]
                    weighted_score += matches[pair] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_score = weighted_score / total_weight
                if combined_score >= threshold:
                    final_matches.append((pair[0], pair[1], float(combined_score)))
        
        return sorted(final_matches, key=lambda x: x[2], reverse=True)
    
    def viewMatches(self, matches, products_a, products_b):
        matched_products = []
        for idx_a, idx_b, score in matches:
            matched_products.append({
                "product_a_name": products_a[idx_a].get("name", ""),
                "product_b_name": products_b[idx_b].get("name", ""),
                "product_a_description": products_a[idx_a].get("features", {}).get("description", ""),
                "product_b_description": products_b[idx_b].get("features", {}).get("description", ""),
                "product_a_category": products_a[idx_a].get("features", {}).get("category", ""),
                "product_b_category": products_b[idx_b].get("features", {}).get("category", ""),
                "match_score": round(score, 2)
            })
        
        df = pd.DataFrame(matched_products)
        print(f"\nFound {len(matches)} matches using ensemble approach")
        return df