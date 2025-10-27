import boto3
import json
import os
import logging
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class S3Manager:
    """ A class to manage S3 connections and operations. """

    def __init__(self):
        """
            Initialize the S3Manager with configuration from environment variables.
        """

        # Load environment variables from .env file
        load_dotenv()

        # Configuration
        self.LOCAL_DOWNLOAD_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.getenv("LOCAL_DOWNLOAD_FOLDER")
        )

        # S3 Configuration
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
        self.S3_FOLDER = os.getenv("S3_FOLDER")

        # Initialize S3 Client
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv("ACCESS_KEY"),
                aws_secret_access_key=os.getenv("SECRET_KEY"),
                region_name=os.getenv("REGION")
            )
        except ClientError as e:
            logger.error(f"Error initializing S3 client (Check AWS credentials): {e}")

    def _sync_dictionary_files(self):
        """
            Connects to S3, lists all files in the specified prefix, and downloads
            each JSON file to the local dictionary folder.
        """
        logger.info("Starting S3 sync for dictionary files...")

        # Ensure Local Directory Exists
        os.makedirs(self.LOCAL_DOWNLOAD_DIR, exist_ok=True)
        logger.info(f"Local target directory: {self.LOCAL_DOWNLOAD_DIR}")

        # List and Download Files
        try:
            # List all objects within the specified prefix (folder)
            response = self.s3.list_objects_v2(Bucket=self.S3_BUCKET_NAME, Prefix=self.S3_FOLDER)

            if 'Contents' not in response:
                # Use single quotes for the nested os.getenv calls
                logger.error(f"No files found in s3://{self.S3_BUCKET_NAME}/{os.getenv('S3_FOLDER')}")
                return
            
            for obj in response['Contents']:
                s3_key = obj['Key']

                filename = os.path.basename(s3_key)
                local_path = os.path.join(self.LOCAL_DOWNLOAD_DIR, filename)

                # Skip the folder key itself if it exists (e.g., "dictionary/")
                if not s3_key.endswith('.json') or s3_key == self.S3_FOLDER:
                    logger.warning("Skip the folder key itself if it exists")
                    continue

                download_needed = True

                if os.path.exists(local_path):
                    # Get S3 Last Modified Time (from the list_objects_v2 response)
                    s3_last_modified_dt = obj.get('LastModified')
                    
                    # Get Local File Modification Time
                    local_mtime_timestamp = os.path.getmtime(local_path)
                    
                    # Convert S3 DateTime to comparable timestamp
                    s3_mtime_timestamp = s3_last_modified_dt.timestamp()

                    # Compare Timestamps (S3 is authoritative source of truth)
                    # Check if S3 time is greater than local time (i.e., S3 file is newer)
                    if s3_mtime_timestamp <= local_mtime_timestamp:
                        download_needed = False
                        logger.info(f"File {filename} is up-to-date. Skipping download.")
                
                if download_needed:
                    logger.info(f"Downloading {s3_key} (NEW/UPDATED) to {local_path}...")
                    
                    # Download the file
                    self.s3.download_file(self.S3_BUCKET_NAME, s3_key, local_path)
                    logger.info(f"Successfully downloaded {filename}")
                else:
                    logger.debug(f"Skipping {filename} (no update required).")

        except ClientError as e:
            logger.error(f"Error during S3 operation: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

        logger.info("S3 sync completed...")

if __name__ == "__main__":
    s3_manager = S3Manager()
    s3_manager._sync_dictionary_files()
