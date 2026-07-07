import os
import sys
import logging
import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def import_data(dataset_repo: str, token: str) -> bool:
    logger.info(f"Initiating direct file download from private repository: {dataset_repo}")
    
    try:
        # تحميل ملف الـ Parquet مباشرة بناءً على المسار الظاهر في الصورة
        local_parquet_path = hf_hub_download(
            repo_id=dataset_repo,
            filename="data/wikipedia_ar/train.parquet",
            repo_type="dataset",
            token=token
        )
        logger.info("Parquet file successfully downloaded to cache.")
        
        # قراءة البيانات وتحويلها إلى JSON محلي
        output_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(output_dir, exist_ok=True)
        output_json_path = os.path.join(output_dir, "train_corpus.json")
        
        df = pd.read_parquet(local_parquet_path)
        df.to_json(output_json_path, orient="records", force_ascii=False, indent=4)
        
        logger.info(f"Successfully exported data to local destination: {output_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Execution failed during file retrieval: {str(e)}")
        return False

if __name__ == "__main__":
    DATASET_NAME = "palestinian-kg/palestinian-cultural-knowledge"
    
from dotenv import load_dotenv
load_dotenv()

import os
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
  raise ValueError("لازم تحط HF_TOKEN في ملف .env قبل التشغيل")