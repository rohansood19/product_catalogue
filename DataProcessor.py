import pandas as pd


class DataProcessor():

    def load_data(self, paths:list):
        for path in paths:
            df = pd.read_csv(path)
            return df

    def extract_full_text(self, row):
        name = row['name']
        features = row['features']

        if isinstance(features, dict):
            feature_text = " ".join([f"{k}: {str(v)}" for k, v in features.items() if pd.notna(v)])
        else:
            feature_text = str(features)  

        return f"{name} {feature_text}".strip()

    def select_columns(self, data, columns: list):
        # |||| Note: specify the columns to select for your specific use case and dataset ||||
        existing_columns = [col for col in columns if col in data.columns]
        return data[existing_columns]

    def process_data(self, df, mode):
        # |||| Note: This is specific to the data provided - you have to modify to fit your data ||||
        result = []
        if mode == "ts":
            for _, row in df.iterrows():
                product = {
                    "name": row.get("name"),
                    "features": {
                        "slug": row.get("slug"),
                        "url": row.get("url"),
                        "category": row.get("category"),
                        "parent_category": row.get("parent_category"),
                        "domain": row.get("domain"),
                        "filled_description": row.get("filled_description")
                    }
                }
                result.append(product)

        elif mode == "bd":
            for _, row in df.iterrows():
                product = {
                    "name": row.get("product_name"),
                    "features": {
                        "description": row.get("description"),
                        "seller_website": row.get("seller_website"),
                        "main_category": row.get("main_category"),
                        "overview": row.get("overview"),
                        "categories": row.get("categories")
                    }
                }
                result.append(product)

        else:
            raise ValueError("Invalid mode. Use 'ts' or 'bd'.")

        return result


