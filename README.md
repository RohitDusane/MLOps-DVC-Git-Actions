# MLops - USing Github Actions

## Built and Setup up files and Project structure
1. Build a virtual enviornment
2. Create a setup.py and requirementst.txt files
3. Logging & Exception.py files
4. Create a GCP Bucket - for data storage and access.
5. Link service account and key with GCP Bucket
6. Initiate data ingestion
    > Update configuration.yaml, path_config.py, data_ingestion.py
    > set GOOGLE_APPLICATION_CREDENTIALS=path to ADC key.json

7. Data Processing/Transformation
    > Update configuration.yaml, path_config.py, data_preprocessing.py

8. Build and Train Model
    > Update configuration.yaml, path_config.py, data_preprocessing.py