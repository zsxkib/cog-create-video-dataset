# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import shutil
import subprocess
import torch
import logging
import warnings
from zipfile import ZipFile
from typing import Optional, Set, List, Union, Tuple
from yt_dlp import YoutubeDL
from cog import BasePredictor, Input, Path
import datetime
import math
import json
import numpy as np
import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from PIL import Image, ImageDraw, ImageFont
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")
for logger in ["torch", "transformers", "cog", "PIL"]:
    logging.getLogger(logger).setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# Constants for Qwen2-VL model
QWEN_MODEL_CACHE = "qwen_checkpoints"
QWEN_MODEL_URL = (
    "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the QWEN model for auto-captioning and verify ffmpeg"""
        # Verify ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "ffmpeg is required but not found. Please install ffmpeg."
            )

        # Download and setup QWEN model
        if not os.path.exists(QWEN_MODEL_CACHE):
            print(f"Downloading Qwen2-VL model to {QWEN_MODEL_CACHE}")
            try:
                subprocess.check_call(["pget", "-xf", QWEN_MODEL_URL, QWEN_MODEL_CACHE])
            except subprocess.CalledProcessError as e:
                print(f"Error downloading model: {str(e)}")
                print("Will attempt to continue if model already exists partially...")

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            print("\nLoading QWEN model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_CACHE,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)
            print("✅ QWEN model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading QWEN model: {str(e)}")
            print("Will continue with limited functionality (no auto-captioning)")
            self.model = None
            self.processor = None

        # Create output directories in the current working directory
        self.temp_dir = "video_processing"
        self.videos_dir = os.path.join(self.temp_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Convert filename to a safe version without spaces or special characters"""
        # Remove file extension if present
        base_name = os.path.splitext(filename)[0]

        # Replace spaces and special characters
        safe_name = base_name.lower()  # Convert to lowercase
        safe_name = safe_name.replace(" - ", "_")  # Handle common separator
        safe_name = safe_name.replace("-", "_")  # Replace dashes
        safe_name = "".join(
            c for c in safe_name if c.isalnum() or c in "_"
        )  # Keep only alphanumeric and underscore

        # Remove tmp and videos from path components if present
        safe_name = safe_name.replace("tmpvideo_processingvideos", "")

        # Ensure it doesn't start with a number
        if safe_name[0].isdigit():
            safe_name = "v_" + safe_name

        return safe_name

    def download_video(self, url: str, output_dir: str) -> str:
        """Download video from URL"""
        ydl_opts = {
            # Get best quality available but prefer h264 codec for compatibility
            "format": "bestvideo[vcodec^=avc]+bestaudio[acodec^=mp4a]/best[ext=mp4]/best",
            # Use MP4 container
            "merge_output_format": "mp4",
            # Keep original quality when possible
            "postprocessor_args": ["-c:v", "copy", "-c:a", "copy"],
            # Quality verification logging
            "quiet": False,
            "no_warnings": False,
            "verbose": True,
            # Ensure we get .mp4 extension
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "progress_hooks": [self._download_progress_hook],
        }

        with YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            print("\n📊 Available Video Formats:")
            info = ydl.extract_info(url, download=False)

            # Print all available formats
            if "formats" in info:
                print("\nAvailable formats:")
                for f in info["formats"]:
                    if "height" in f and "ext" in f:
                        print(
                            f"Format {f['format_id']}: {f['ext']} - {f.get('height', 'N/A')}p"
                        )

            # Now download best quality
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # Print selected format details
            print(f"\n📊 Selected Format:")
            if "resolution" in info:
                print(f"Resolution: {info['resolution']}")
            if "fps" in info:
                print(f"FPS: {info['fps']}")
            if "vcodec" in info:
                print(f"Video Codec: {info['vcodec']}")
            if "acodec" in info:
                print(f"Audio Codec: {info['acodec']}")

            # Ensure we have the correct extension
            if not filename.endswith(".mp4"):
                filename = os.path.splitext(filename)[0] + ".mp4"

            # Create sanitized filename
            sanitized_name = f"{self.sanitize_filename(os.path.splitext(os.path.basename(filename))[0])}.mp4"
            new_path = os.path.join(output_dir, sanitized_name)

            # Move file if needed
            if os.path.exists(filename) and filename != new_path:
                shutil.move(filename, new_path)
                print(f"\n📝 Renamed file to: {sanitized_name}")

            return sanitized_name

    def _download_progress_hook(self, d):
        """Custom progress display for downloads"""
        if d["status"] == "downloading":
            try:
                percent = float(d.get("_percent_str", "0%").replace("%", ""))
                if percent % 10 == 0:  # Only show every 10%
                    print(f"\rDownloading... {percent:.0f}%", end="", flush=True)
            except:
                pass
        elif d["status"] == "finished":
            print("\rDownload completed ✓")

    def detect_scenes(
        self, 
        video_path: str, 
        detection_mode: str = "content",
        min_scene_length: float = 1.0,
        threshold: float = None,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> List[Tuple]:
        """
        Detect scenes in a video using the specified detection method.
        
        Args:
            video_path: Path to video file
            detection_mode: Detection method ('content', 'adaptive', or 'threshold')
            min_scene_length: Minimum scene length in seconds
            threshold: Detection threshold (if None, use default for the detector)
            start_time: Start time in seconds for scene detection
            end_time: End time in seconds for scene detection (0 = until end)
            
        Returns:
            List of scene tuples (start_time, end_time)
        """
        print(f"\n🔍 Detecting scenes using {detection_mode.upper()} detector...")
        
        # Open video using the modern API
        video = open_video(video_path)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        
        # Add the appropriate detector
        if detection_mode == "content":
            # ContentDetector is best for most fast-cut content
            detector = ContentDetector(
                threshold=threshold if threshold else 27.0,
                min_scene_len=min_scene_length
            )
        elif detection_mode == "adaptive":
            # AdaptiveDetector handles camera movement better
            # Note: AdaptiveDetector doesn't accept threshold parameter
            detector = AdaptiveDetector(
                min_scene_len=min_scene_length
            )
        elif detection_mode == "threshold":
            # ThresholdDetector is best for fade in/out transitions
            detector = ThresholdDetector(
                threshold=threshold if threshold else 12.0,
                min_scene_len=min_scene_length
            )
        else:
            raise ValueError(f"Unknown detection mode: {detection_mode}")
        
        scene_manager.add_detector(detector)
        
        # Set time range if specified
        kwargs = {}
        if start_time > 0 and end_time > 0:
            if start_time >= end_time:
                raise ValueError("End time must be greater than start time")
            kwargs["start_time"] = start_time
            kwargs["end_time"] = end_time
        elif start_time > 0:
            kwargs["start_time"] = start_time
        
        # Detect scenes
        scene_manager.detect_scenes(video, show_progress=True, **kwargs)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        print(f"✅ Detected {len(scene_list)} scenes")
        
        # Return the scene list
        return scene_list

    def evaluate_scene_quality(self, video_path: str, scene_list: List, num_scenes: int) -> List:
        """
        Evaluate and select the best scenes based on quality and diversity.
        
        Args:
            video_path: Path to video file
            scene_list: List of detected scenes
            num_scenes: Number of scenes to select (if 0, return all)
            
        Returns:
            Filtered list of selected scenes
        """
        if num_scenes <= 0 or num_scenes >= len(scene_list):
            return scene_list
        
        print(f"\n⭐ Selecting {num_scenes} best scenes from {len(scene_list)} detected scenes...")
        
        cap = cv2.VideoCapture(video_path)
        scene_scores = []
        
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            duration = scene[1].get_seconds() - scene[0].get_seconds()
            
            # Skip very short scenes
            if duration < 0.5:
                scene_scores.append((-1, i))  # Mark for exclusion
                continue
                
            # Sample frames from the scene to evaluate quality
            frames = []
            frame_positions = np.linspace(start_frame, end_frame-1, 5).astype(int)
            
            for pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            if not frames:
                scene_scores.append((-1, i))  # Mark for exclusion
                continue
                
            # Calculate quality metrics
            quality_score = 0
            for frame in frames:
                # Brightness score
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_score = 1.0 if 50 < brightness < 220 else 0.5
                
                # Contrast score
                contrast = np.std(gray)
                contrast_score = min(1.0, contrast / 50.0)
                
                # Blur detection
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_score = min(1.0, laplacian_var / 500.0)
                
                # Combined score
                frame_score = (brightness_score + contrast_score + blur_score) / 3.0
                quality_score += frame_score
            
            quality_score /= len(frames)
            
            # Adjust score based on scene duration (prefer scenes between 2-4 seconds)
            duration_score = 1.0
            if duration < 1.0 or duration > 5.0:
                duration_score = 0.7
            
            # Final score = quality * duration factor
            final_score = quality_score * duration_score
            
            # Store score with scene index
            scene_scores.append((final_score, i))
        
        cap.release()
        
        # Sort scenes by score (descending)
        scene_scores.sort(reverse=True)
        
        # Select top N scenes
        selected_indices = [s[1] for s in scene_scores[:num_scenes] if s[0] > 0]
        
        # Return scenes in chronological order
        selected_indices.sort()
        selected_scenes = [scene_list[i] for i in selected_indices]
        
        print(f"✅ Selected {len(selected_scenes)} high-quality scenes")
        return selected_scenes

    def generate_preview(self, video_path: str, scene_list: List, output_dir: str) -> Path:
        """
        Generate a preview image grid showing selected scenes.
        
        Args:
            video_path: Path to video file
            scene_list: List of selected scenes
            output_dir: Directory to save preview image
            
        Returns:
            Path to the generated preview image
        """
        print("\n🖼️ Generating scene preview...")
        
        # Create a temporary directory for scene images
        preview_dir = os.path.join(output_dir, "preview_images")
        os.makedirs(preview_dir, exist_ok=True)
        
        # Extract center frame from each scene
        video = open_video(video_path)
        print(f"Saving 1 images per scene [format=jpg] {preview_dir}")
        
        try:
            # In newer versions of PySceneDetect, save_images might return different formats
            # Let's handle various potential return types
            image_filenames = save_images(
                scene_list, 
                video,
                num_images=1,  # One image per scene
                image_name_template='scene_{:03d}',  # Simple numbered format
                output_dir=preview_dir
            )
            
            # Debug info about returned filenames
            print(f"✓ Generated scene preview images")
            if image_filenames:
                print(f"  Return type: {type(image_filenames)}")
                if isinstance(image_filenames, list) and image_filenames:
                    print(f"  First item type: {type(image_filenames[0])}")
            
            # Try listing directory to see actual files
            try:
                actual_files = os.listdir(preview_dir)
                print(f"  Found {len(actual_files)} files in preview directory")
                if actual_files:
                    print(f"  First actual file: {actual_files[0]}")
            except Exception as e:
                print(f"  Error checking directory: {str(e)}")
                
        except Exception as e:
            print(f"⚠️ Warning: Error during save_images: {str(e)}")
            image_filenames = []
        
        # Load images
        images = []
        
        # Fallback to directory listing if image_filenames is not what we expect
        if not image_filenames or not isinstance(image_filenames, list):
            print("⚠️ No valid filenames returned from save_images, falling back to directory listing")
            try:
                image_filenames = [f for f in os.listdir(preview_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  Found {len(image_filenames)} image files in directory")
            except Exception as e:
                print(f"  Error listing directory: {str(e)}")
                # If we can't recover, return early
                if not image_filenames:
                    print("❌ Failed to generate preview images")
                    preview_path = os.path.join(output_dir, "scene_preview_failed.txt")
                    with open(preview_path, "w") as f:
                        f.write("Failed to generate preview images\n")
                    return Path(preview_path)
        
        # Ensure filenames are strings before joining paths
        for filename in sorted(image_filenames):
            # Check if filename is not a string and convert if needed
            if not isinstance(filename, (str, bytes, os.PathLike)):
                # Convert to string if it's another type (like int)
                filename = str(filename)
            
            img_path = os.path.join(preview_dir, filename)
            try:
                img = Image.open(img_path)
                # Resize to a reasonable size
                img = img.resize((320, 180))
                images.append(img)
            except Exception as e:
                print(f"⚠️ Warning: Could not load image {filename}: {str(e)}")
                # Continue with other images
        
        # Check if we have any images
        if not images:
            print("❌ No preview images could be loaded")
            preview_path = os.path.join(output_dir, "scene_preview_failed.txt")
            with open(preview_path, "w") as f:
                f.write("Failed to load any preview images\n")
            return Path(preview_path)
        
        # Calculate grid dimensions
        num_scenes = len(scene_list)
        grid_cols = min(5, num_scenes)
        grid_rows = math.ceil(num_scenes / grid_cols)
        
        # Create empty grid
        cell_width, cell_height = 320, 220  # Image height + space for text
        grid_width = cell_width * grid_cols
        grid_height = cell_height * grid_rows
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(grid_image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial", 12)
        except:
            font = ImageFont.load_default()
        
        # Place images and metadata on grid
        for i, (img, scene) in enumerate(zip(images, scene_list)):
            row = i // grid_cols
            col = i % grid_cols
            
            # Calculate position
            x = col * cell_width
            y = row * cell_height
            
            # Paste image
            grid_image.paste(img, (x, y))
            
            # Add scene info
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time
            
            draw.text(
                (x + 5, y + 185),
                f"Scene {i+1}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)",
                fill=(255, 255, 255),
                font=font
            )
        
        # Save grid image
        preview_path = os.path.join(output_dir, "scene_preview.jpg")
        grid_image.save(preview_path, quality=95)
        
        print(f"✅ Preview image saved to: {preview_path}")
        print(f"   Individual frame images saved in: {preview_dir}")
        return Path(preview_path)

    def autocaption_videos(
        self,
        video_files: Set[str],
        caption_files: Set[str],
        trigger_word: Optional[str] = None,
        autocaption_prefix: Optional[str] = None,
        autocaption_suffix: Optional[str] = None,
    ) -> None:
        """Generate captions for videos that don't have matching .txt files."""
        videos_without_captions = video_files - caption_files
        if not videos_without_captions:
            return

        print("\n=== 🤖 Auto-captioning Videos ===")
        print(f"Found {len(videos_without_captions)} videos without captions")

        for i, vid_name in enumerate(videos_without_captions, 1):
            mp4_path = os.path.join(self.videos_dir, vid_name + ".mp4")
            if os.path.exists(mp4_path):
                print(f"\n[{i}/{len(videos_without_captions)}] 🎥 {vid_name}.mp4")

                # Use absolute path
                abs_path = os.path.abspath(mp4_path)

                # Build caption components
                prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                trigger = f"{trigger_word} " if trigger_word else ""

                # Prepare messages format with customized prompt
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": abs_path,
                            },
                            {
                                "type": "text",
                                "text": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                            },
                        ],
                    }
                ]

                try:
                    # Process inputs
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    # Import qwen utils here to avoid circular imports
                    from qwen_vl_utils import process_vision_info

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")

                    print("\nGenerating caption...")
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    caption = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    # Combine prefix, trigger, caption, and suffix
                    final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                    print("\n📝 Generated Caption:")
                    print("--------------------")
                    print(f"{final_caption}")
                    print("--------------------")

                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to autocaption {vid_name}.mp4")
                    print(f"Error: {str(e)}")
                    final_caption = (
                        f"{prefix}{trigger}A video clip named {vid_name}{suffix}"
                    )
                    print("\n📝 Using fallback caption:")
                    print("--------------------")
                    print(f"{final_caption}")
                    print("--------------------")

                # Save caption
                txt_path = os.path.join(self.videos_dir, vid_name + ".txt")
                with open(txt_path, "w") as f:
                    f.write(final_caption.strip() + "\n")

                print(f"✅ Saved to: {txt_path}")

        print(f"\n✨ Successfully processed {len(videos_without_captions)} videos!")
        print("=====================================")

    def generate_caption(
        self,
        video_path: str,
        prompt: str = "Describe this video clip briefly, focusing on the main action and visual elements.",
        trigger_word: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        """Generate caption for a video using QWEN-VL"""
        # Check if model is available
        if self.model is None or self.processor is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            prefix_text = f"{prefix.strip()} " if prefix else ""
            suffix_text = f" {suffix.strip()}" if suffix else ""
            trigger = f"{trigger_word} " if trigger_word else ""
            return f"{prefix_text}{trigger}A video clip named {video_name}{suffix_text}"
        
        try:
            # Build caption components
            prefix_text = f"{prefix.strip()} " if prefix else ""
            suffix_text = f" {suffix.strip()}" if suffix else ""
            trigger = f"{trigger_word} " if trigger_word else ""

            # Prepare messages format with custom prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": os.path.abspath(video_path),
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                print("\n⚠️ Warning: Failed to import qwen_vl_utils")
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                return f"{prefix_text}{trigger}A video clip named {video_name}{suffix_text}"

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Generate caption (increased token limit)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Combine all components
            return f"{prefix_text}{trigger}{caption.strip()}{suffix_text}"

        except Exception as e:
            print(f"\n⚠️ Warning: Failed to generate caption")
            print(f"Error: {str(e)}")
            # Fallback caption
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            return f"{prefix_text}{trigger}A video clip named {video_name}{suffix_text}"

    def process_video_segment(
        self, 
        input_path: str, 
        output_path: str,
        start_time: float,
        duration: float,
        quality: str = "balanced"
    ) -> bool:
        """
        Process a single video segment with quality settings.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            start_time: Start time in seconds
            duration: Duration in seconds
            quality: Quality preset ('fast', 'balanced', 'high')
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Quality preset parameters
        quality_presets = {
            "fast": {
                "crf": "28",
                "preset": "ultrafast"
            },
            "balanced": {
                "crf": "22",
                "preset": "veryfast"
            },
            "high": {
                "crf": "17",
                "preset": "slow"
            }
        }
        
        if quality not in quality_presets:
            quality = "balanced"
            
        preset = quality_presets[quality]
        
        try:
            # First, try with seeking before input
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-ss",
                    str(start_time),
                    "-i",
                    input_path,
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-crf",
                    preset["crf"],
                    "-preset",
                    preset["preset"],
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    output_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            
            # Verify the output file exists and has non-zero size
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise subprocess.CalledProcessError(1, "ffmpeg", "Output file is empty")
                
            return True
            
        except Exception as e:
            print(f"Failed to create segment: {str(e)}")
            return False

    def extract_scenes_from_video(
        self,
        input_path: str,
        scene_list: List,
        output_dir: str,
        quality: str = "balanced",
        max_workers: int = 4
    ) -> List[str]:
        """
        Extract scenes from a video into separate clips.
        
        Args:
            input_path: Path to input video
            scene_list: List of detected scenes
            output_dir: Output directory for clips
            quality: Quality preset ('fast', 'balanced', 'high')
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of paths to extracted clips
        """
        base_name = self.sanitize_filename(os.path.basename(input_path))
        output_files = []
        
        print(f"\n✂️ Extracting {len(scene_list)} scenes...")
        
        # Prepare tasks for parallel processing
        tasks = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time
            
            output_path = f"{output_dir}/{base_name}_scene{i+1:02d}.mp4"
            
            tasks.append((
                input_path,
                output_path,
                start_time,
                duration,
                quality
            ))
            
            # Store output path for return
            output_files.append(output_path)
            
        # Process scenes in parallel
        successful = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_video_segment, *task) for task in tasks]
            
            for i, future in enumerate(futures):
                try:
                    if future.result():
                        successful += 1
                        print(f"  ✓ Extracted scene {i+1}/{len(scene_list)}")
                    else:
                        print(f"  ❌ Failed to extract scene {i+1}/{len(scene_list)}")
                except Exception as e:
                    print(f"  ❌ Error extracting scene {i+1}: {str(e)}")
        
        print(f"\n✅ Successfully extracted {successful}/{len(scene_list)} scenes")
        
        # Only return paths to successful extractions
        return [f for f in output_files if os.path.exists(f)]

    def predict(
        self,
        video_url: str = Input(
            description="YouTube/video URL to process. Leave empty if uploading a file. Note: URL takes precedence if both URL and file are provided.",
            default=None,
        ),
        video_file: Path = Input(
            description="Video file to process. Leave empty if using URL. Ignored if URL is provided.",
            default=None,
        ),
        # Scene detection options
        detection_mode: str = Input(
            description="Scene detection method: 'content' (fast cuts), 'adaptive' (camera movement), or 'threshold' (fades)",
            default="content",
            choices=["content", "adaptive", "threshold"]
        ),
        min_scene_length: float = Input(
            description="Minimum scene length in seconds", 
            default=1.0
        ),
        max_scene_length: float = Input(
            description="Maximum scene length in seconds", 
            default=5.0
        ),
        num_scenes: int = Input(
            description="Number of scenes to extract (0 = all detected scenes)",
            default=4,
        ),
        # Time selection 
        target_fps: float = Input(
            description="Target frame rate (e.g. 24, 30). Set to -1 to keep original fps.",
            default=24.0,
        ),
        start_time: float = Input(
            description="Start time in seconds for video processing",
            default=0.0,
        ),
        end_time: float = Input(
            description="End time in seconds for video processing. Set to 0 to process until the end.",
            default=0.0,
        ),
        skip_intro: bool = Input(
            description="Automatically skip first 10 seconds (typical intro)",
            default=False,
        ),
        # Output options
        preview_only: bool = Input(
            description="Generate scene previews without creating full dataset",
            default=False,
        ),
        quality: str = Input(
            description="Video quality preset: 'fast' (lower quality, smaller files), 'balanced', or 'high' (best quality, larger files)",
            default="balanced",
            choices=["fast", "balanced", "high"]
        ),
        # Captioning options
        autocaption: bool = Input(
            description="Let AI generate a caption for your video. If False, you must provide custom_caption.",
            default=True,
        ),
        caption_style: str = Input(
            description="Caption style: 'minimal' (short), 'detailed' (longer descriptions), or 'custom'",
            default="detailed",
            choices=["minimal", "detailed", "custom"]
        ),
        custom_caption: str = Input(
            description="Your custom caption. Required if caption_style is 'custom' or autocaption is False.",
            default=None,
        ),
        trigger_word: str = Input(
            description="Trigger word to include in captions (e.g., TOK, STYLE3D). Will be added at start of caption.",
            default="TOK",
        ),
        autocaption_prefix: str = Input(
            description="Text to add BEFORE caption. Example: 'a video of'",
            default=None,
        ),
        autocaption_suffix: str = Input(
            description="Text to add AFTER caption. Example: 'in a cinematic style'",
            default=None,
        ),
    ) -> List[Path]:
        """Process a video, detect scenes, and create a training-ready dataset with captions."""

        print("\n=======================================")
        print("🎬 VIDEO DATASET CREATOR: PROCESSING START")
        print("=======================================")
        
        # Clean up from previous runs at the start
        print("\n🧹 Cleaning up previous run data...")
        
        # Check if directories exist before cleaning
        if os.path.exists(self.temp_dir):
            # Remove video files
            if os.path.exists(self.videos_dir):
                for file in os.listdir(self.videos_dir):
                    try:
                        file_path = os.path.join(self.videos_dir, file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error removing file {file}: {e}")
            
            # Remove preview images directory
            preview_dir = os.path.join(self.temp_dir, "preview_images")
            if os.path.exists(preview_dir):
                try:
                    for file in os.listdir(preview_dir):
                        file_path = os.path.join(preview_dir, file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                except Exception as e:
                    print(f"Error removing preview images: {e}")
            
            # Remove other files in temp_dir (but keep the directories)
            for file in os.listdir(self.temp_dir):
                try:
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
        
        print("✅ Cleanup complete")
        
        try:
            # Input validation
            if video_url and video_file:
                print(
                    "\n⚠️ Warning: Both URL and file provided. Using URL and ignoring file."
                )
            elif not video_url and not video_file:
                raise ValueError("Must provide either video_url or video_file")

            # Apply skip intro if requested
            if skip_intro and start_time == 0.0:
                start_time = 10.0
                print(f"\n🎬 Skipping first 10 seconds (intro)")

            # Create working directory
            temp_dir = self.temp_dir
            videos_dir = self.videos_dir
            os.makedirs(videos_dir, exist_ok=True)

            # Step 1: Video Input
            print("\n[STEP 1/5] 📥 ACQUIRING VIDEO")
            
            # Handle video input
            if video_url:
                print(f"Downloading video from: {video_url}")
                filename = self.download_video(video_url, videos_dir)
                video_path = os.path.join(videos_dir, filename)
            else:
                print(f"Processing uploaded video: {video_file.name}")
                sanitized_name = f"{self.sanitize_filename(video_file.name)}.mp4"
                video_path = os.path.join(videos_dir, sanitized_name)
                shutil.copy(str(video_file), video_path)
                filename = sanitized_name

            # Step 2: FPS Conversion (if needed)
            print("\n[STEP 2/5] 🎥 PREPARING VIDEO")
            
            # Apply fps conversion if needed
            if target_fps > 0:
                print(f"Converting frame rate to {target_fps} fps...")
                temp_path = video_path + ".temp.mp4"
                
                # Get original fps
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of",
                        "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True, check=True
                )
                num, den = map(int, probe.stdout.strip().split('/'))
                original_fps = num / den
                print(f"Original fps: {original_fps:.2f}")
                
                # Convert fps while maintaining quality
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={target_fps}",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-c:a", "copy", temp_path],
                    capture_output=True, text=True, check=True
                )
                
                # Verify the conversion
                verify = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", temp_path],
                    capture_output=True, text=True, check=True
                )
                
                info = json.loads(verify.stdout)
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        print(f"\n✓ Converted Video Quality:")
                        print(f"   Resolution: {stream.get('width')}x{stream.get('height')}")
                        fps_str = stream.get('r_frame_rate', '0/1')
                        num, den = map(int, fps_str.split('/'))
                        print(f"   Frame Rate: {num/den:.2f} fps")
                        break
                
                os.replace(temp_path, video_path)
                print(f"✅ Successfully converted frame rate")
            else:
                # Display video info even if not converting
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", video_path],
                    capture_output=True, text=True, check=True
                )
                
                info = json.loads(probe.stdout)
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        print(f"\n✓ Video Information:")
                        print(f"   Resolution: {stream.get('width')}x{stream.get('height')}")
                        fps_str = stream.get('r_frame_rate', '0/1')
                        num, den = map(int, fps_str.split('/'))
                        print(f"   Frame Rate: {num/den:.2f} fps")
                        break

            # Step 3: Scene Detection
            print(f"\n[STEP 3/5] 🔍 DETECTING SCENES")
            print(f"Using {detection_mode.upper()} detection mode with {min_scene_length}s minimum scene length")
            
            try:
                # Detect scenes in the video
                scene_list = self.detect_scenes(
                    video_path,
                    detection_mode=detection_mode,
                    min_scene_length=min_scene_length,
                    start_time=start_time,
                    end_time=end_time if end_time > 0 else 0.0
                )
                
                # Filter scenes based on max_scene_length
                if max_scene_length > 0:
                    original_count = len(scene_list)
                    scene_list = [scene for scene in scene_list 
                                if (scene[1].get_seconds() - scene[0].get_seconds()) <= max_scene_length]
                    print(f"📊 Filtered from {original_count} to {len(scene_list)} scenes within {max_scene_length}s max length")
            except Exception as e:
                print(f"\n⚠️ Error during scene detection: {str(e)}")
                print("Retrying with content detector as fallback...")
                try:
                    # Fallback to content detector with default settings
                    scene_list = self.detect_scenes(
                        video_path,
                        detection_mode="content",
                        min_scene_length=min_scene_length,
                        start_time=start_time,
                        end_time=end_time if end_time > 0 else 0.0
                    )
                except Exception as e2:
                    print(f"\n❌ Failed with fallback detector: {str(e2)}")
                    raise RuntimeError(f"Scene detection failed: {str(e)}. Fallback also failed: {str(e2)}")
            
            # If no scenes were detected
            if not scene_list:
                print("\n⚠️ No scenes detected with current settings. Try adjusting detection parameters.")
                # Suggestions to help user
                print("\nSuggested fixes:")
                print(f"- Decrease min_scene_length (current: {min_scene_length}s)")
                print(f"- Try a different detection_mode (current: {detection_mode})")
                print("- Check if video has distinct scene changes")
                
                error_path = os.path.join(self.temp_dir, "no_scenes_detected.txt")
                with open(error_path, "w") as f:
                    f.write("No scenes detected with current settings.\n")
                    f.write(f"min_scene_length: {min_scene_length}\n")
                    f.write(f"detection_mode: {detection_mode}\n")
                
                return [Path(error_path)]
            
            # Select best scenes if requested
            if num_scenes > 0 and num_scenes < len(scene_list):
                print(f"🔍 Selecting {num_scenes} best scenes from {len(scene_list)} detected scenes")
                scene_list = self.evaluate_scene_quality(video_path, scene_list, num_scenes)
            
            # Step 4: Generate Preview (skip if in preview_only mode)
            if not preview_only:
                print(f"\n[STEP 4/5] 🖼️ GENERATING SCENE PREVIEW")
                preview_path = self.generate_preview(video_path, scene_list, temp_dir)
            else:
                # Skip preview generation in preview_only mode
                print(f"\n[STEP 4/5] 👉 SKIPPING PREVIEW GENERATION IN PREVIEW MODE")
                preview_path = None
            
            # Step 5: Extract Scenes & Create Videos (do this even in preview mode)
            print(f"\n[STEP 5/5] ✂️ EXTRACTING SCENES")
            print(f"Video quality preset: {quality}")
            
            # Extract scenes from the video
            extracted_clips = self.extract_scenes_from_video(
                video_path,
                scene_list,
                videos_dir,
                quality=quality
            )
            
            if not extracted_clips:
                print("\n⚠️ No clips were successfully extracted. Check for ffmpeg errors.")
                error_path = os.path.join(self.temp_dir, "no_clips_extracted.txt")
                with open(error_path, "w") as f:
                    f.write("No clips were successfully extracted. Check for ffmpeg errors.\n")
                
                return [Path(error_path)]
            
            # If preview only mode, return the extracted clips now
            if preview_only:
                print("\n🎬 PREVIEW MODE ACTIVE - RETURNING EXTRACTED CLIPS")
                
                print(f"\n✨ Success! Extracted {len(extracted_clips)} video clips")
                for i, clip in enumerate(extracted_clips):
                    print(f"  Clip {i+1}: {os.path.basename(clip)}")
                
                # Convert all paths to Path objects - but ONLY return the MP4 clips
                clip_paths = [Path(clip) for clip in extracted_clips]
                
                print(f"To create the full dataset with captions, run again without preview_only=true")
                
                return clip_paths
            
            # Continue with captioning if not in preview mode
            print(f"\n[STEP 6/6] 📝 ADDING CAPTIONS")
            
            # Generate captions based on caption style
            prompt_templates = {
                "minimal": "Briefly describe what's happening in this video clip in a short sentence.",
                "detailed": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                "custom": custom_caption
            }
            
            if caption_style == "custom" and not custom_caption:
                caption_style = "detailed"
                print("\n⚠️ No custom caption provided. Using 'detailed' style instead.")
            
            # Check if autocaption is disabled but no custom caption provided
            if not autocaption and not custom_caption:
                print("\n⚠️ Autocaption is disabled but no custom caption provided.")
                print("Using basic caption with filename.")
                # Create a simple caption template based on filename
                custom_caption = "A video clip from the dataset"
            
            caption_prompt = prompt_templates[caption_style]
            print(f"\n📝 Using {caption_style.upper()} caption style")
            print(f"👤 Autocaption: {'Enabled' if autocaption else 'Disabled'}")
            
            # Get video name and caption files
            video_files = set()
            caption_files = set()
            
            for clip_path in extracted_clips:
                base_name = os.path.splitext(os.path.basename(clip_path))[0]
                video_files.add(base_name)
                
                # Create caption for this clip
                print(f"\n🎬 Processing clip: {base_name}")
                
                if autocaption:
                    # Use AI to generate caption
                    final_caption = self.generate_caption(
                        clip_path,
                        prompt=caption_prompt,
                        trigger_word=trigger_word,
                        prefix=autocaption_prefix,
                        suffix=autocaption_suffix,
                    )
                else:
                    # Use provided custom caption or fallback
                    prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                    suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                    trigger = f"{trigger_word} " if trigger_word else ""
                    final_caption = f"{prefix}{trigger}{custom_caption.strip()}{suffix}"
                
                # Save caption
                txt_path = os.path.join(videos_dir, f"{base_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_caption.strip() + "\n")
                    
                caption_files.add(base_name)
                
                print(f"\n📝 Caption for {base_name}:")
                print("--------------------")
                print(final_caption)
                print("--------------------")
                print(f"✅ Saved to: {txt_path}")

            # Create zip file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"video_dataset_{timestamp}.zip"

            with ZipFile(output_path, "w") as zipf:
                print(f"\n📦 Creating zip file...")
                for clip_path in extracted_clips:
                    base_name = os.path.splitext(os.path.basename(clip_path))[0]
                    zipf.write(clip_path, f"videos/{os.path.basename(clip_path)}")
                    txt_path = os.path.join(videos_dir, f"{base_name}.txt")
                    zipf.write(txt_path, f"videos/{base_name}.txt")
                    
                # Add preview image to zip
                if preview_path:
                    zipf.write(preview_path, "scene_preview.jpg")
                    
                # Add readme with usage info
                with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as readme:
                    readme.write("VIDEO DATASET CREATED WITH COG-CREATE-VIDEO-DATASET\n")
                    readme.write("=================================================\n\n")
                    readme.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    readme.write(f"Source: {video_url if video_url else video_file.name}\n")
                    readme.write(f"Scenes: {len(extracted_clips)}\n")
                    readme.write(f"Caption Style: {caption_style}\n\n")
                    readme.write("This dataset is ready for use in video generation model training.\n")
                    readme.write("The 'videos' directory contains MP4 clips and matching TXT captions.\n")
                
                zipf.write(readme.name, "README.txt")
                os.unlink(readme.name)

            # Show zip contents
            with ZipFile(output_path, "r") as zipf:
                zip_info = zipf.infolist()

            print("\n📋 Zip contents:")
            print("--------------------")
            print(f"{'Size':>10}  {'Name'}")
            print(f"{'----':>10}  {'----'}")
            for info in zip_info:
                size_str = (
                    f"{info.file_size/1024/1024:.1f}M"
                    if info.file_size > 1024 * 1024
                    else f"{info.file_size/1024:.1f}K"
                )
                print(f"{size_str:>10}  {info.filename}")
            print("--------------------")

            print(f"\n✨ Success! Dataset created with {len(extracted_clips)} video clips")
            print(f"Output saved to: {output_path}")
            
            print("\n=======================================")
            print("🎬 VIDEO DATASET CREATOR: COMPLETE")
            print("=======================================")
                
            return [Path(output_path)]
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            print("\nPlease check your input parameters and try again.")
            print("If the problem persists, try with different scene detection settings.")
            
            # Create error report
            error_path = "dataset_creation_error.txt"
            with open(error_path, "w") as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"Parameters:\n")
                f.write(f"- video_url: {video_url}\n")
                f.write(f"- detection_mode: {detection_mode}\n")
                f.write(f"- min_scene_length: {min_scene_length}\n")
                f.write(f"- max_scene_length: {max_scene_length}\n")
                f.write(f"- num_scenes: {num_scenes}\n")
                
            return [Path(error_path)]

