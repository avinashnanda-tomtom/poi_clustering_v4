from azure.storage.blob import BlobClient

connection_string = "DefaultEndpointsProtocol=https;AccountName=poimachinelearning;AccountKey=p4EEQkYTnq4jkfVtyAC2iKhVwGRSP96AumqVC8YXAzxU6h3r3Ns/5L5FuKqm3R4WtxgfPyOhfC+3lGnSgimQSA==;EndpointSuffix=core.windows.net"
blob = BlobClient.from_connection_string(
    conn_str=connection_string,
    container_name="poi-clustering-models",
    blob_name="model_NZL_stage2.txt",
)

with open(
    "/workspace/clustering/Entity_matching/model_lgb_em/model_NZL_stage2.txt", "rb"
) as data:
    blob.upload_blob(data)
