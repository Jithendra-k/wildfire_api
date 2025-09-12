from google.cloud import storage

bucket_name = "data_housee"

client = storage.Client()
bucket = client.bucket(bucket_name)

# List first 5 blobs in the wildfire folder
blobs = list(bucket.list_blobs(prefix="wildfire_ml_data/featured_data_with_risk_parquet/"))

for blob in blobs[:5]:
    print(blob.name)
