import boto3
import pandas as pd

class DataLoader:
    
    def getKeys(self):
        # Replace relative import with absolute import
        import settings
        return settings.access["ACCESS_KEY"], settings.access["SECRET_ACCESS_KEY"]
    
    def getData(self):
        access_key, secret_access_key = self.getKeys()
        s3_bucket = boto3.Session(access_key, secret_access_key, region_name='us-east-1')
        #----{ Connect to S3 client }------
        s3 = s3_bucket.client('s3')

        object_key = "ts_technologies.csv"
        bucket_name = "product-catalogue-report"
        #----{ Download file to local }------
        s3.download_file(bucket_name, object_key, 'ts_technologies.csv' )
        #----{ Read JSON data }------
        with open('ts_technologies.csv', 'r') as auth_json:
            data1 = pd.read_csv(auth_json)

        object_key = "bd_technologies.csv"

        #----{ Download file to local }------
        s3.download_file(bucket_name, object_key, 'bd_technologies.csv' )
        #----{ Read JSON data }------
        with open('bd_technologies.csv', 'r') as auth_json:
            data2 = pd.read_csv(auth_json)
        
        return data1, data2
        
    def peek(self, data, mode):
        if mode == "shape":
            print(data.shape)
        elif mode == "head":
            print(data.head(3))
        elif mode == "columns_names":
            print(data.columns.tolist())
        elif mode == "info":
            print(data.isnull().sum())
            print(data.duplicated().sum())
            print(data.info())
            
    def addFlags(self, data1, data2):

        data1['description_missing'] = data1['description'].isnull()
        data1['url_missing'] = data1['url'].isnull()

        data2['seller_website_missing'] = data2['seller_website'].isnull()
        data2['categories_missing'] = data2['categories'].isnull()
        data2['headquarters_missing'] = data2['headquarters'].isnull()
        
        return data1, data2

    def extract_domain(self, url):
        from urllib.parse import urlparse
        if pd.isna(url) or not isinstance(url, str) or url.strip() == "":
            return ""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace("www.", "")
        return domain
    
    def urlExtractor(self, data1, data2):
        data1['domain'] = data1['url'].apply(self.extract_domain)
        data2['domain'] = data2['seller_website'].apply(self.extract_domain)
        data1_domains = data1[['url', 'domain']].dropna().head(5)
        data2_domains = data2[['seller_website', 'domain']].dropna().head(5)
        print("Data1 Sample:")
        print(data1_domains)
        print("Data2 Sample:")
        print(data2_domains)
        return data1, data2

if __name__ == "__main__":
    d = DataLoader()
    data1, data2 = d.getData()
    data1, data2 = d.addFlags(data1, data2)
    print("Missing flags sample for data1:")
    print(data1)
    print("Missing flags sample for data2:")
    print(data2)
    data1, data2 = d.urlExtractor(data1, data2)
    print(data1.head(3))
    print(data2.head(3))