import requests
from dotenv import load_dotenv
import os
import argparse
import json

"""
A command-line interface to access the backend of remote VLM server
- Usage:
    python3 main.py \
        --file_path {file_path} \
        --system_prompt {system_prompt} \
        --message_prompt {message_prompt} \
        --max_new_tokens {max_new_tokens} \
"""

def main():

    # Command line parser
    parser = argparse.ArgumentParser(description='Send image to VLM backend for analysis')
    parser.add_argument(
        '--file_path',
        type=str,
        required=True,
        help='Path to the image file'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default='',
        help='System prompt for the model'
    )
    parser.add_argument(
        '--message_prompt',
        type=str,
        default='',
        help='Message prompt for the model'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    args = parser.parse_args()
    
    # Load URL
    load_dotenv()
    url = os.getenv("URL")
    if not url:
        print("Error: URL not found in environment variables")
        return
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return
    
    # Send request
    try:
        with open(args.file_path, "rb") as f:
            files = {
                "image": (os.path.basename(args.file_path), f, "image/png")
            }
            data = {
                "system_prompt": args.system_prompt,
                "message_prompt": args.message_prompt,
                "max_new_tokens": str(args.max_new_tokens)
            }
            response = requests.post(url, files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("\nResult:")
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.text}")

    # Handle other errors
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()