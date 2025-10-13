#!/bin/bash
set -e

echo "Starting dictionary synchronization..."
# Ensure the LOCAL_DOWNLOAD_FOLDER (e.g., 'dictionary') is created or exists.
# The sync_dictionary.py script will handle the actual content, but this ensures the folder exists if needed.
mkdir -p /app/dictionary

# Run the S3 sync script. 
# We use the full path relative to the WORKDIR (/app)
python3 python3 config/sync_dicts.py

echo "Dictionary synchronization complete."

# Execute the main command passed to the container (the Gunicorn CMD)
exec "$@" 