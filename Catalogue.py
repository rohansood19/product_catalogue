import pandas as pd


class Consolidator():

  def select_and_rename_columns(self, data, column_map: dict):
    existing_map = {new_col: old_col for new_col, old_col in column_map.items() if old_col in data.columns}
    return data[list(existing_map.values())].rename(columns={v: k for k, v in existing_map.items()})
  
  def processor(self, matchData, generated_data1, data2):
    df = pd.read_csv(matchData)
    ts_df = pd.read_csv(generated_data1)
    bd_df = pd.read_csv(data2)

    bd_df_unique = bd_df[['product_name', 'seller_website']].drop_duplicates(subset='product_name')
    merged_df = df.merge(bd_df_unique, how='left', left_on='product_b_name', right_on='product_name')

    del df
    del bd_df_unique

    print("Matched Items: ", len(merged_df))

    filtered_df_ts = ts_df[~ts_df['name'].isin(merged_df['product_a_name'])]
    filtered_df_bd = bd_df[~bd_df['product_name'].isin(merged_df['product_b_name'])]

    del ts_df
    del bd_df

    print("Data1 Items: ", len(filtered_df_ts))
    print("Data2 Items: ", len(filtered_df_bd))

    column_map_ts = {
    "name": "name",
    "description": "filled_description",
    "url": "url"
    }

    column_map_bd = {
        "name": "product_name",
        "description": "description",
        "url": "seller_website"
        }

    column_map_matches = {"name": "product_b_name",
                          "description": "product_b_description",
                          "url": "seller_website"
                          }

    filtered_df_ts = self.select_and_rename_columns(filtered_df_ts, column_map_ts)
    filtered_df_bd = self.select_and_rename_columns(filtered_df_bd, column_map_bd)
    filtered_df_matches = self.select_and_rename_columns(merged_df, column_map_matches)

    del merged_df

    return filtered_df_ts, filtered_df_bd, filtered_df_matches

  def masterCatalogueGenerator(self, filtered_df_ts, filtered_df_bd, filtered_df_matches):
    print("Generating Unmatched Unique Data")
    unique_data = pd.concat([filtered_df_ts, filtered_df_bd]).drop_duplicates(subset=['name'], keep='first')
    print("Unique Items: ", len(unique_data))
    del filtered_df_ts
    del filtered_df_bd
    print("Generating Master Catalogue")
    consolidated_df = pd.concat([filtered_df_matches, unique_data])
    del unique_data
    print("Master Catalogue Items: ", len(consolidated_df))
    return consolidated_df