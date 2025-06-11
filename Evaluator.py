import pandas as pd
import numpy as np
from itertools import combinations

class ModelEvaluation:
    def __init__(self):
        self.model_results = {}
        self.model_names = ['faiss', 'semantic', 'tfidf', 'bert', 'ensemble']  
    
    def load_results(self, faiss_path: str, semantic_path: str, tfidf_path: str, 
                    bert_path: str, ensemble_path: str):
        """Load results from all models including ensemble"""
        self.model_results = {
            'faiss': pd.read_csv(faiss_path),
            'semantic': pd.read_csv(semantic_path),
            'tfidf': pd.read_csv(tfidf_path),
            'bert': pd.read_csv(bert_path),
            'ensemble': pd.read_csv(ensemble_path)  
        }
    
    def get_matching_statistics(self):
        """Calculate basic statistics about matches for each model"""
        stats = {}
        for model_name, df in self.model_results.items():
            stats[model_name] = {
                "total_matches": len(df),
                "avg_score": df['match_score'].mean(),
                "min_score": df['match_score'].min(),
                "max_score": df['match_score'].max(),
                "std_score": df['match_score'].std()
            }
        return stats
    
    def compare_all_models(self):
        """Compare matches found by all models"""
        model_pairs = {}
        for model_name, df in self.model_results.items():
            model_pairs[model_name] = set(zip(df['product_a_name'], df['product_b_name']))
        
        comparisons = {}
        # Compare each pair of models
        for model1, model2 in combinations(self.model_names, 2):
            common = len(model_pairs[model1].intersection(model_pairs[model2]))
            only_model1 = len(model_pairs[model1] - model_pairs[model2])
            only_model2 = len(model_pairs[model2] - model_pairs[model1])
            total = len(model_pairs[model1].union(model_pairs[model2]))
            
            comparisons[f"{model1}_vs_{model2}"] = {
                "common_matches": common,
                "only_" + model1: only_model1,
                "only_" + model2: only_model2,
                "agreement_ratio": common / total if total > 0 else 0
            }
        
        # Find matches common to all models
        common_to_all = set.intersection(*model_pairs.values())
        comparisons["all_models"] = {
            "common_matches": len(common_to_all),
            "agreement_ratio": len(common_to_all) / len(set.union(*model_pairs.values()))
        }
        
        return comparisons
    
    def analyze_category_matches(self):
        """Analyze matches based on product categories for all models"""
        category_accuracy = {}
        for model_name, df in self.model_results.items():
            category_matches = (df['product_a_category'] == df['product_b_category']).mean()
            category_accuracy[model_name] = category_matches
        return category_accuracy
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        stats = self.get_matching_statistics()
        comparisons = self.compare_all_models()
        
        # Create comparison matrix
        comparison_data = []
        metrics = ["Total Matches", "Average Score"]
        
        for model in self.model_names:
            row = [
                stats[model]["total_matches"],
                round(stats[model]["avg_score"], 3)
            ]
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data, 
                                   columns=metrics,
                                   index=self.model_names)
        
        # Create agreement matrix
        agreement_matrix = pd.DataFrame(index=self.model_names, 
                                      columns=self.model_names,
                                      dtype=float)
        
        for model1, model2 in combinations(self.model_names, 2):
            agreement = comparisons[f"{model1}_vs_{model2}"]["agreement_ratio"]
            agreement_matrix.loc[model1, model2] = agreement
            agreement_matrix.loc[model2, model1] = agreement
        
        # Fill diagonal with 1.0
        for model in self.model_names:
            agreement_matrix.loc[model, model] = 1.0
        
        return {
            "model_comparison": df_comparison,
            "agreement_matrix": agreement_matrix,
            "common_to_all": comparisons["all_models"]
        }
