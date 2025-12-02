"""
Groq API Caption Generator Module

This module handles caption generation using Groq API with the 
meta-llama/llama-4-maverick-17b-128e-instruct model for generating
detailed facial descriptions.
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import json

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq package not installed. Install with: pip install groq")


class GroqCaptionGenerator:
    """
    Caption generator using Groq API for facial image descriptions.
    
    Designed for distributed processing across multiple team members
    to handle large datasets efficiently.
    """
    
    # The exact caption prompt from the paper
    CAPTION_PROMPT = """Provide a detailed physical description of the person in this image. Include the following attributes: approximate age, gender, ethnicity/race, facial expression and emotion, hair characteristics (color, length, style, hairline), facial features (eyes, nose, mouth, lips, chin, cheekbones), facial hair if present, accessories (eyeglasses, hats, jewelry), body type indicators (chubby, bags under eyes), skin tone, makeup if visible, and any other distinctive physical characteristics. Describe whether the mouth is open or closed, and the state of the eyes. Write the description as a single flowing short paragraph."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        """
        Initialize the Groq caption generator.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var
            model: Model name to use (default: llama-4-maverick-17b-128e-instruct)
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq package is required. Install with: pip install groq")
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.request_count = 0
        self.rate_limit_delay = 0.1  # Small delay to avoid rate limits
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_caption(self, image_path: str, retry_count: int = 3) -> str:
        """
        Generate a caption for a single image.
        
        Args:
            image_path: Path to the image file
            retry_count: Number of retry attempts on failure
            
        Returns:
            Generated caption text
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Make API request with retries
        for attempt in range(retry_count):
            try:
                # Add small delay to respect rate limits
                if self.request_count > 0:
                    time.sleep(self.rate_limit_delay)
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.CAPTION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model=self.model,
                    temperature=1.0,
                    max_completion_tokens=1024,
                    top_p=1.0,
                    stream=False,
                )
                
                self.request_count += 1
                caption = chat_completion.choices[0].message.content.strip()
                return caption
                
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate caption after {retry_count} attempts: {e}")
        
        raise Exception("Caption generation failed")
    
    def batch_generate_captions(
        self,
        image_paths: List[str],
        output_file: Optional[str] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, str]]:
        """
        Generate captions for a batch of images.
        
        Designed for distributed processing - each team member can process
        a subset of images by specifying start_index and end_index.
        
        Args:
            image_paths: List of image file paths
            output_file: Optional JSONL file to save results incrementally
            start_index: Starting index for this batch (for distributed processing)
            end_index: Ending index for this batch (None = process all)
            progress_callback: Optional callback function(current, total, caption)
            
        Returns:
            List of dictionaries with 'file_name' and 'text' keys
        """
        if end_index is None:
            end_index = len(image_paths)
        
        image_paths = image_paths[start_index:end_index]
        results = []
        
        print(f"Processing {len(image_paths)} images (indices {start_index} to {end_index-1})...")
        
        for idx, image_path in enumerate(image_paths):
            try:
                image_name = Path(image_path).name
                print(f"[{idx+1}/{len(image_paths)}] Processing: {image_name}")
                
                caption = self.generate_caption(image_path)
                
                result = {
                    "file_name": image_name,
                    "text": caption
                }
                results.append(result)
                
                # Save incrementally if output file specified
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # Progress callback
                if progress_callback:
                    progress_callback(start_index + idx + 1, len(image_paths), caption)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Continue with next image
                continue
        
        print(f"Completed: {len(results)}/{len(image_paths)} captions generated")
        return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python groq_caption_generator.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        generator = GroqCaptionGenerator()
        caption = generator.generate_caption(image_path)
        print("\n" + "="*80)
        print("GENERATED CAPTION:")
        print("="*80)
        print(caption)
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed groq: pip install groq")
        print("2. Set GROQ_API_KEY environment variable")

