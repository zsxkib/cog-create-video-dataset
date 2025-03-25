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
        """Load the QWEN model for auto-captioning"""
        # Create output directories
        self.temp_dir = "video_processing"
        self.videos_dir = os.path.join(self.temp_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # Download QWEN model if needed
        if not os.path.exists(QWEN_MODEL_CACHE):
            print(f"Downloading Qwen2-VL model to {QWEN_MODEL_CACHE}")
            subprocess.run(["pget", "-xf", QWEN_MODEL_URL, QWEN_MODEL_CACHE])

        # Import and load model
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_CACHE,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)
            print("✅ QWEN model loaded for auto-captioning")
        except Exception as e:
            print(f"⚠️ Auto-captioning disabled: {str(e)}")
            self.model = None
            self.processor = None

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
        if threshold is not None:
            print(f"Using threshold value: {threshold}")
        
        # Open video using the modern API
        video = open_video(video_path)
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        
        # Add the appropriate detector
        if detection_mode == "adaptive":
            # AdaptiveDetector handles camera movement better
            detector_threshold = threshold if threshold is not None else 3.0
            detector = AdaptiveDetector(min_scene_len=min_scene_length, threshold=detector_threshold)
        elif detection_mode == "threshold":
            # ThresholdDetector is best for fade in/out transitions
            detector_threshold = threshold if threshold is not None else 12.0
            detector = ThresholdDetector(threshold=detector_threshold, min_scene_len=min_scene_length)
        else:
            # ContentDetector is best for most fast-cut content and is default
            detector_threshold = threshold if threshold is not None else 27.0
            detector = ContentDetector(threshold=detector_threshold, min_scene_len=min_scene_length)
        
        scene_manager.add_detector(detector)
        
        # Set time range if specified
        kwargs = {}
        if start_time > 0:
            kwargs["start_time"] = start_time
            if end_time > start_time:
                kwargs["end_time"] = end_time
        
        # Detect scenes
        scene_manager.detect_scenes(video, show_progress=True, **kwargs)
        scene_list = scene_manager.get_scene_list()
        
        print(f"✅ Detected {len(scene_list)} scenes")
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
        
        # Skip if no scenes detected
        if not scene_list:
            print("❌ No scenes to preview")
            preview_path = os.path.join(output_dir, "scene_preview_empty.txt")
            with open(preview_path, "w") as f:
                f.write("No scenes detected to preview\n")
            return Path(preview_path)
        
        # Extract center frame from each scene
        video = open_video(video_path)
        print(f"Saving 1 image per scene [format=jpg] to {preview_dir}")
        
        # Extract frame images
        image_filenames = save_images(
            scene_list, 
            video,
            num_images=1,  # One image per scene
            image_name_template='scene_{:03d}',  # Simple numbered format
            output_dir=preview_dir
        )
        
        # Find image files in directory
        actual_files = os.listdir(preview_dir)
        image_paths = []
        
        # Find all images in the directory
        for filename in sorted(actual_files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(preview_dir, filename)
                img = Image.open(img_path)
                img = img.resize((320, 180))  # Resize for grid
                image_paths.append((img_path, img))
        
        # Calculate grid dimensions
        num_images = len(image_paths)
        grid_cols = min(5, num_images)
        grid_rows = math.ceil(num_images / grid_cols)
        
        # Create empty grid
        cell_width, cell_height = 320, 220  # Image height + space for text
        grid_width = cell_width * grid_cols
        grid_height = cell_height * grid_rows
        
        grid_image = Image.new('RGB', (grid_width, grid_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(grid_image)
        
        # Use default font
        font = ImageFont.load_default()
        
        # Place images and metadata on grid
        for i, (_, img) in enumerate(image_paths[:len(scene_list)]):
            scene = scene_list[i]
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
        # Format components
        prefix_text = f"{prefix.strip()} " if prefix else ""
        suffix_text = f" {suffix.strip()}" if suffix else ""
        trigger = f"{trigger_word} " if trigger_word else ""
        
        # Fallback caption (used if model not available or on error)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fallback_caption = f"{prefix_text}{trigger}A video clip named {video_name}{suffix_text}"
        
        # Check if model is available
        if self.model is None or self.processor is None:
            return fallback_caption
        
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

        # Import vision utility
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare model inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate caption
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

        # Return formatted caption
        return f"{prefix_text}{trigger}{caption.strip()}{suffix_text}"

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
            # Extract segment with ffmpeg
            subprocess.run(
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
                check=True,
            )
            
            # Verify output exists
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            return False
            
        except:
            # Simple failure handling
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
                if future.result():
                    successful += 1
                    print(f"  ✓ Extracted scene {i+1}/{len(scene_list)}")
                else:
                    print(f"  ❌ Failed to extract scene {i+1}/{len(scene_list)}")
        
        print(f"\n✅ Successfully extracted {successful}/{len(scene_list)} scenes")
        
        # Only return paths to successful extractions
        return [f for f in output_files if os.path.exists(f)]

    def predict(
        self,
        video_url: Optional[str] = Input(
            description="YouTube/video URL to process. Leave empty if uploading a file. Note: URL takes precedence if both URL and file are provided.",
            default=None,
        ),
        video_file: Optional[Path] = Input(
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
            default=10.0
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
        custom_caption: Optional[str] = Input(
            description="Your custom caption. Required if caption_style is 'custom' or autocaption is False.",
            default=None,
        ),
        trigger_word: str = Input(
            description="Trigger word to include in captions (e.g., TOK, STYLE3D). Will be added at start of caption.",
            default="TOK",
        ),
        autocaption_prefix: Optional[str] = Input(
            description="Text to add BEFORE caption. Example: 'a video of'",
            default=None,
        ),
        autocaption_suffix: Optional[str] = Input(
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
        self._cleanup_workspace()
        
        # Input validation and setup
        if video_url and video_file:
            print("\n⚠️ Warning: Both URL and file provided. Using URL and ignoring file.")
        elif not video_url and not video_file:
            return self._create_error_report("Error: Must provide either video_url or video_file")

        # Apply skip intro if requested
        if skip_intro and start_time == 0.0:
            start_time = 10.0
            print(f"\n🎬 Skipping first 10 seconds (intro)")

        # Step 1: Video acquisition
        video_path = self._acquire_video(video_url, video_file)
        
        # Step 2: FPS conversion if needed
        video_path = self._prepare_video(video_path, target_fps)
        
        # Step 3: Scene detection
        scene_list = self._detect_and_filter_scenes(
            video_path, 
            detection_mode, 
            min_scene_length, 
            max_scene_length, 
            start_time, 
            end_time, 
            num_scenes
        )
        
        # Check if we have valid scenes
        if not scene_list:
            return self._no_scenes_detected_error(min_scene_length, detection_mode)
        
        # Step 4: Generate preview (optional) - REMOVED as per user request
        print(f"\n[STEP 4/5] 👉 SKIPPING PREVIEW GENERATION")
        
        # Step 5: Extract scenes
        print(f"\n[STEP 5/5] ✂️ EXTRACTING SCENES")
        print(f"Video quality preset: {quality}")
        
        extracted_clips = self.extract_scenes_from_video(
            video_path,
            scene_list,
            self.videos_dir,
            quality=quality
        )
        
        if not extracted_clips:
            return self._create_error_report("No clips were successfully extracted. Check for ffmpeg errors.")
        
        # Step 6: Add captions (always do this regardless of preview_only)
        print(f"\n[STEP 6/6] 📝 ADDING CAPTIONS")
        
        # Determine if we should skip captions
        skip_captions = not autocaption and not custom_caption
        if skip_captions:
            print("\n⚠️ Autocaption is disabled and no custom caption provided.")
            print("Skipping caption generation - no text files will be included.")
        else:
            self._generate_captions(
                extracted_clips, 
                autocaption, 
                custom_caption, 
                caption_style, 
                trigger_word, 
                autocaption_prefix, 
                autocaption_suffix
            )
        
        # Create zip file (always do this, but only return it if not in preview mode)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_dataset_{timestamp}.zip"
        
        self._create_output_zip(
            output_path, 
            extracted_clips, 
            skip_captions, 
            None,  # No preview path 
            video_url, 
            video_file, 
            caption_style
        )
        
        # Return appropriate output based on preview_only mode
        if preview_only:
            print("\n🎬 PREVIEW MODE ACTIVE - RETURNING EXTRACTED CLIPS")
            print(f"\n✨ Success! Extracted {len(extracted_clips)} video clips")
            for i, clip in enumerate(extracted_clips):
                print(f"  Clip {i+1}: {os.path.basename(clip)}")
            
            # Convert clips to Path objects
            clip_paths = [Path(clip) for clip in extracted_clips]
            
            print(f"To use the full dataset with captions, look in: {output_path}")
            return clip_paths
        else:
            print(f"\n✨ Success! Dataset created with {len(extracted_clips)} video clips")
            print(f"Output saved to: {output_path}")
            
            print("\n=======================================")
            print("🎬 VIDEO DATASET CREATOR: COMPLETE")
            print("=======================================")
                
            return [Path(output_path)]
    
    def _cleanup_workspace(self):
        """Clean up previous run files"""
        if os.path.exists(self.temp_dir):
            # Remove video files
            if os.path.exists(self.videos_dir):
                for file in os.listdir(self.videos_dir):
                    file_path = os.path.join(self.videos_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            
            # Remove preview images directory
            preview_dir = os.path.join(self.temp_dir, "preview_images")
            if os.path.exists(preview_dir):
                for file in os.listdir(preview_dir):
                    file_path = os.path.join(preview_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            
            # Remove other files in temp_dir (but keep directories)
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        print("✅ Cleanup complete")
    
    def _acquire_video(self, video_url, video_file):
        """Download or copy video to workspace"""
        print("\n[STEP 1/5] 📥 ACQUIRING VIDEO")
        
        # Make sure videos directory exists
        os.makedirs(self.videos_dir, exist_ok=True)
        
        if video_url:
            print(f"Downloading video from: {video_url}")
            filename = self.download_video(video_url, self.videos_dir)
            return os.path.join(self.videos_dir, filename)
        else:
            print(f"Processing uploaded video: {video_file.name}")
            sanitized_name = f"{self.sanitize_filename(video_file.name)}.mp4"
            video_path = os.path.join(self.videos_dir, sanitized_name)
            shutil.copy(str(video_file), video_path)
            return video_path
    
    def _prepare_video(self, video_path, target_fps):
        """Convert video FPS if needed and get video info"""
        print("\n[STEP 2/5] 🎥 PREPARING VIDEO")
        
        # Display video info
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", video_path],
            capture_output=True, text=True, check=True
        )
        
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get('r_frame_rate', '0/1')
                num, den = map(int, fps_str.split('/'))
                original_fps = num / den
                
                print(f"\n✓ Original Video:")
                print(f"   Resolution: {stream.get('width')}x{stream.get('height')}")
                print(f"   Frame Rate: {original_fps:.2f} fps")
                break
        
        # Apply fps conversion if needed
        if target_fps > 0:
            print(f"Converting frame rate to {target_fps} fps...")
            temp_path = video_path + ".temp.mp4"
            
            # Convert fps while maintaining quality
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={target_fps}",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-c:a", "copy", temp_path],
                capture_output=True, text=True, check=True
            )
            
            os.replace(temp_path, video_path)
            print(f"✅ Successfully converted frame rate")
            
        return video_path
    
    def _detect_and_filter_scenes(self, video_path, detection_mode, min_scene_length, 
                                   max_scene_length, start_time, end_time, num_scenes):
        """Detect scenes and filter based on parameters"""
        print(f"\n[STEP 3/5] 🔍 DETECTING SCENES")
        print(f"Using {detection_mode.upper()} detection mode with {min_scene_length}s minimum scene length")
        
        # Try primary detection mode first, with a fallback only if needed
        detection_modes_to_try = [detection_mode]
        if detection_mode != "content":
            detection_modes_to_try.append("content")  # Add content as fallback
        
        scene_list = []
        
        # Default thresholds for each mode
        default_thresholds = {
            "content": 27.0,
            "threshold": 12.0,
            "adaptive": 3.0
        }
        
        for mode in detection_modes_to_try:
            threshold = default_thresholds.get(mode)
            
            try:
                if mode != detection_mode:
                    print(f"\n⚠️ Trying {mode.upper()} detector with threshold={threshold}...")
                
                scene_list = self.detect_scenes(
                    video_path,
                    detection_mode=mode,
                    min_scene_length=min_scene_length,
                    threshold=threshold,
                    start_time=start_time,
                    end_time=end_time if end_time > 0 else 0.0
                )
                
                if scene_list and (num_scenes <= 0 or len(scene_list) >= num_scenes):
                    print(f"✅ Found sufficient scenes with {mode} detector")
                    break  # Success with enough scenes
            except Exception as e:
                print(f"⚠️ Scene detection failed with {mode} detector: {str(e)}")
                scene_list = []  # Reset scene list
        
        # Create evenly spaced scenes if detection failed
        if not scene_list:
            print("\n⚠️ Scene detection failed. Creating evenly spaced scenes instead.")
            scene_list = self._create_evenly_spaced_scenes(
                video_path, 
                min_scene_length=min_scene_length,
                max_scene_length=max_scene_length,
                start_time=start_time,
                end_time=end_time,
                num_scenes=num_scenes if num_scenes > 0 else 4  # Default to 4 scenes if not specified
            )
        
        # Apply max_scene_length filter
        if max_scene_length > 0 and scene_list:
            original_count = len(scene_list)
            filtered_scenes = [scene for scene in scene_list 
                              if (scene[1].get_seconds() - scene[0].get_seconds()) <= max_scene_length]
            
            # Only apply filter if we'll still have enough scenes
            if num_scenes <= 0 or len(filtered_scenes) >= num_scenes:
                scene_list = filtered_scenes
                print(f"📊 Filtered from {original_count} to {len(scene_list)} scenes within {max_scene_length}s max length")
        
        # Select best scenes if we have more scenes than needed
        if scene_list and num_scenes > 0:
            if len(scene_list) > num_scenes:
                print(f"🔍 Selecting {num_scenes} best scenes from {len(scene_list)} detected scenes")
                scene_list = self.evaluate_scene_quality(video_path, scene_list, num_scenes)
            elif len(scene_list) < num_scenes:
                print(f"\n⚠️ Only detected {len(scene_list)} scenes, fewer than the requested {num_scenes}.")
            
        return scene_list
    
    def _create_evenly_spaced_scenes(self, video_path, min_scene_length=1.0, max_scene_length=5.0, 
                                    start_time=0.0, end_time=0.0, num_scenes=4):
        """Create evenly spaced scenes when scene detection fails"""
        from scenedetect.frame_timecode import FrameTimecode
        
        print(f"📏 Creating {num_scenes} evenly spaced scenes...")
        
        # Open video to get duration and fps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate total duration
        total_duration = frame_count / fps
        
        # Adjust for start and end times
        if end_time <= 0 or end_time > total_duration:
            end_time = total_duration
            
        if start_time >= end_time:
            start_time = 0
        
        usable_duration = end_time - start_time
        
        # Calculate scene duration
        scene_duration = min(max_scene_length, max(min_scene_length, usable_duration / num_scenes))
        
        # Create scene list
        scene_list = []
        for i in range(num_scenes):
            scene_start = start_time + (i * scene_duration)
            scene_end = min(scene_start + scene_duration, end_time)
            
            # Convert to frame numbers
            start_frame = int(scene_start * fps)
            end_frame = int(scene_end * fps)
            
            # Create FrameTimecode objects
            start_tc = FrameTimecode(start_frame, fps)
            end_tc = FrameTimecode(end_frame, fps)
            
            scene_list.append((start_tc, end_tc))
            
        print(f"✅ Created {len(scene_list)} evenly spaced scenes")
        return scene_list
    
    def _no_scenes_detected_error(self, min_scene_length, detection_mode):
        """Handle the case where no scenes were detected"""
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
    
    def _generate_captions(self, extracted_clips, autocaption, custom_caption, 
                         caption_style, trigger_word, autocaption_prefix, autocaption_suffix):
        """Generate captions for video clips"""
        # Generate captions based on caption style
        prompt_templates = {
            "minimal": "Briefly describe what's happening in this video clip in a short sentence.",
            "detailed": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
            "custom": custom_caption
        }
        
        if caption_style == "custom" and not custom_caption:
            caption_style = "detailed"
            print("\n⚠️ No custom caption provided. Using 'detailed' style instead.")
        
        caption_prompt = prompt_templates[caption_style]
        print(f"\n📝 Using {caption_style.upper()} caption style")
        print(f"👤 Autocaption: {'Enabled' if autocaption else 'Disabled'}")
        
        for clip_path in extracted_clips:
            base_name = os.path.splitext(os.path.basename(clip_path))[0]
            
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
                # Use provided custom caption
                prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                trigger = f"{trigger_word} " if trigger_word else ""
                final_caption = f"{prefix}{trigger}{custom_caption.strip()}{suffix}"
            
            # Save caption
            txt_path = os.path.join(self.videos_dir, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_caption.strip() + "\n")
                
            print(f"\n📝 Caption for {base_name}:")
            print("--------------------")
            print(final_caption)
            print("--------------------")
            print(f"✅ Saved to: {txt_path}")
    
    def _create_output_zip(self, output_path, extracted_clips, skip_captions, 
                           preview_path, video_url, video_file, caption_style):
        """Create ZIP file with dataset contents"""
        print(f"\n📦 Creating zip file...")
        
        # Check if an old preview file exists and remove it to avoid including it in the zip
        default_preview_path = os.path.join(self.temp_dir, "scene_preview.jpg")
        if os.path.exists(default_preview_path):
            print(f"Removing unused preview image: {default_preview_path}")
            os.remove(default_preview_path)
        
        with ZipFile(output_path, "w") as zipf:
            # Add video clips
            for clip_path in extracted_clips:
                base_name = os.path.splitext(os.path.basename(clip_path))[0]
                zipf.write(clip_path, f"videos/{os.path.basename(clip_path)}")
                
                # Only include text files if captions weren't skipped
                if not skip_captions:
                    txt_path = os.path.join(self.videos_dir, f"{base_name}.txt")
                    zipf.write(txt_path, f"videos/{base_name}.txt")
            
            # Add readme with usage info
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as readme:
                readme.write("VIDEO DATASET CREATED WITH COG-CREATE-VIDEO-DATASET\n")
                readme.write("=================================================\n\n")
                readme.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                readme.write(f"Source: {video_url if video_url else video_file.name}\n")
                readme.write(f"Scenes: {len(extracted_clips)}\n")
                
                if skip_captions:
                    readme.write(f"Caption Style: None (No captions included)\n\n")
                else:
                    readme.write(f"Caption Style: {caption_style}\n\n")
                
                readme.write("This dataset is ready for use in video generation model training.\n")
                if skip_captions:
                    readme.write("The 'videos' directory contains MP4 clips only (no captions).\n")
                else:
                    readme.write("The 'videos' directory contains MP4 clips and matching TXT captions.\n")
            
            zipf.write(readme.name, "README.txt")
            os.unlink(readme.name)

        # Display zip contents
        self._show_zip_contents(output_path)
    
    def _show_zip_contents(self, zip_path):
        """Display contents of the created ZIP file"""
        with ZipFile(zip_path, "r") as zipf:
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
    
    def _create_error_report(self, error_message, **params):
        """Create error report and return path"""
        print(f"\n❌ ERROR: {error_message}")
        print("\nPlease check your input parameters and try again.")
        
        error_path = "dataset_creation_error.txt"
        with open(error_path, "w") as f:
            f.write(f"Error: {error_message}\n")
            if params:
                f.write(f"Parameters:\n")
                for key, value in params.items():
                    f.write(f"- {key}: {value}\n")
        
        return [Path(error_path)]

