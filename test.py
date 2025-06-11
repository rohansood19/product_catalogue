if __name__ == "__main__":
    import os
    import json
    from datetime import datetime
    os.makedirs('catalogue', exist_ok=True)
    from DataLoaders import *
    from GenerateDescription import *
    from Models import *
    from DataProcessor import *
    from Evaluator import *
    from Catalogue import *
    
    # Load data
    data_loader = DataLoader()
    gen = GenerateDescription()
    processor = DataProcessor()
    modelFaiss = ModelFaiss()
    modelSemantic = ModelSemanticSearch()
    modelTFIDF = ModelTFIDF(threshold=0.8, batch_size=100)
    modelBert = ModelBERT(model_name='sentence-transformers/all-distilroberta-v1', threshold=0.8, batch_size=100)
    modelEnsemble = ModelEnsemble(weights={
                                            'faiss': 0.45,      # High weight due to reliability
                                            'semantic': 0.50,   # Equal to FAISS as they're highly correlated
                                            'bert': 0.05,       # For diversity and to catch unique matches
                                        })
    evaluator = ModelEvaluation()
    consol = Consolidator()
    
    data1, data2 = data_loader.getData()
    data1, data2 = data_loader.addFlags(data1, data2)
    print("Missing flags sample for data1:")
    print(data1)
    print("Missing flags sample for data2:")
    print(data2)
    data1, data2 = data_loader.urlExtractor(data1, data2)
    print(data1.head(3))
    print(data2.head(3))
    
    model_name = "t5-small"
    tokenizer, model = gen.generator(model_name)
    data1 = gen.generate(data1, model, tokenizer, save_path='ts_df_gen.csv')
    
    columns_data1 = ["name", "slug", "url", "category", "parent_category", "domain", "filled_description"]
    columns_data2 = ["product_name", "description", "seller_website", "main_category", "overview", "categories"]
    
    ts_df = processor.load_data(['ts_df_gen.csv'])
    bd_df = processor.load_data(['bd_technologies.csv'])
    ts_df = processor.select_columns(ts_df, columns_data1)
    bd_df = processor.select_columns(bd_df, columns_data2)
    ts_products = processor.process_data(ts_df, mode='ts')
    bd_products = processor.process_data(bd_df, mode='bd')
    
    matches_faiss = modelFaiss.get_matched_products(ts_products, bd_products, model='all-MiniLM-L6-v2', threshold=0.8)
    df_matches_faiss = modelFaiss.viewMatches(matches_faiss, ts_products, bd_products)
    with open('matches_faiss.csv', 'w') as f:
        df_matches_faiss.to_csv(f, index=False)
    
    matches_semantic = modelSemantic.get_matched_products(ts_products, bd_products, model='all-MiniLM-L6-v2', threshold=0.8)
    df_matches_semantic = modelSemantic.viewMatches(matches_semantic, ts_products, bd_products)
    with open('matches_semantic.csv', 'w') as f:
        df_matches_semantic.to_csv(f, index=False)
        
    matches_tfidf = modelTFIDF.get_matched_products(ts_products, bd_products)
    df_matches_tfidf = modelTFIDF.viewMatches(matches_tfidf, ts_products, bd_products)
    with open('matches_tfidf.csv', 'w') as f:
        df_matches_tfidf.to_csv(f, index=False)
    
    matches = modelBert.get_matched_products(ts_products, bd_products)
    results_df = modelBert.viewMatches(matches, ts_products, bd_products)
    with open('matches_bert.csv', 'w') as f:
        results_df.to_csv(f, index=False)
    
    matches_ensemble = modelEnsemble.get_matched_products(ts_products, bd_products)
    df_matches_ensemble = modelEnsemble.viewMatches(matches_ensemble, ts_products, bd_products)
    with open('matches_ensemble.csv', 'w') as f:
        df_matches_ensemble.to_csv(f, index=False)
    
    # Update evaluator to include ensemble results
    evaluator.load_results(
            faiss_path="matches_faiss.csv",
            semantic_path="matches_semantic.csv",
            tfidf_path="matches_tfidf.csv",
            bert_path="matches_bert.csv",
            ensemble_path="matches_ensemble.csv"
        )

    report = evaluator.generate_report()

    print("\nModel Comparison (Including Ensemble):")
    print(report["model_comparison"].to_string())

    print("\nModel Agreement Matrix (Including Ensemble):")
    print(report["agreement_matrix"].to_string())

    print("\nCommon to All Models (Including Ensemble):")
    print(f"Matches found by all models: {report['common_to_all']['common_matches']}")
    print(f"Overall agreement ratio: {report['common_to_all']['agreement_ratio']:.2%}")

    # Get detailed comparisons
    comparisons = evaluator.compare_all_models()
    print("\nPairwise Comparisons with Ensemble:")
    for pair, metrics in comparisons.items():
        if pair != "all_models":
            print(f"\n{pair}:")
            print(f"Common matches: {metrics['common_matches']}")
            print(f"Agreement ratio: {metrics['agreement_ratio']:.2%}")
    
    # Additional ensemble-specific metrics
    print("\nEnsemble Model Performance:")
    print(f"Total matches found by ensemble: {len(df_matches_ensemble)}")
    print(f"Average confidence score: {df_matches_ensemble['match_score'].mean():.3f}")
    print(f"Category match rate: {(df_matches_ensemble['product_a_category'] == df_matches_ensemble['product_b_category']).mean():.2%}")
    
    models_data = {
    'faiss': 'matches_faiss.csv',
    'semantic': 'matches_semantic.csv',
    'tfidf': 'matches_tfidf.csv',
    'bert': 'matches_bert.csv',
    'ensemble': 'matches_ensemble.csv'
    }

    catalogues = {}
    print("\nGenerating Catalogues for Each Model:")
    for model_name, matches_file in models_data.items():
        print(f"\nProcessing {model_name.upper()} Model Catalogue:")
        processed_data1, processed_data2, processed_matches = consol.processor(
            matchData=matches_file,
            generated_data1="ts_df_gen.csv",
            data2="bd_technologies.csv"
        )
        
        master_catalogue = consol.masterCatalogueGenerator(
            processed_data1,
            processed_data2,
            processed_matches
        )
        
        # Save catalogue to specific directory
        catalogue_path = f'catalogue/{model_name}'
        os.makedirs(catalogue_path, exist_ok=True)
        master_catalogue.to_csv(f'{catalogue_path}/master_catalogue.csv', index=False)
        catalogues[model_name] = len(master_catalogue)
        print(f"{model_name.capitalize()} Catalogue saved with {len(master_catalogue)} entries")
    