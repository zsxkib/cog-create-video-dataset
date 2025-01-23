# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import shutil
import subprocess
import torch
import logging
import warnings
from zipfile import ZipFile
from typing import Optional, Set
from yt_dlp import YoutubeDL
from cog import BasePredictor, Input, Path
import datetime
import math
import json

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
            subprocess.check_call(["pget", "-xf", QWEN_MODEL_URL, QWEN_MODEL_CACHE])

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print("\nLoading QWEN model...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)

        # Create temporary directories
        self.temp_dir = "/tmp/video_processing"
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
        """Download video and return sanitized filename"""
        ydl_opts = {
            "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",  # Prefer MP4 format
            "merge_output_format": "mp4",  # Ensure final output is MP4
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "progress_hooks": [self._download_progress_hook],
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # Get the original downloaded file path
            original_path = os.path.join(output_dir, filename)

            # Create sanitized filename
            sanitized_name = f"{self.sanitize_filename(filename)}.mp4"
            new_path = os.path.join(output_dir, sanitized_name)

            # Rename the file
            if original_path != new_path:
                shutil.move(original_path, new_path)
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

            from qwen_vl_utils import process_vision_info

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
            print(f"\n⚠️  Warning: Failed to generate caption")
            print(f"Error: {str(e)}")
            # Fallback caption
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            return f"{prefix_text}{trigger}A video clip named {video_name}{suffix_text}"

    def split_video(
        self, input_path: str, start_time: float, duration: float, num_segments: int
    ) -> list[str]:
        """Split video into equal segments

        Args:
            input_path: Path to input video
            start_time: Start time in seconds
            duration: Duration in seconds to use from start time
            num_segments: Number of segments to split into

        Returns:
            List of paths to generated segment files
        """
        # Get video info
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                input_path,
            ],
            capture_output=True,
            text=True,
        )
        video_info = json.loads(probe.stdout)
        total_duration = float(video_info["format"]["duration"])

        # Validate start_time and duration
        if start_time < 0:
            raise ValueError("start_time cannot be negative")
        if start_time >= total_duration:
            raise ValueError(
                f"start_time ({start_time}s) exceeds video duration ({total_duration}s)"
            )

        # Calculate effective duration
        max_duration = total_duration - start_time
        effective_duration = min(duration, max_duration)
        segment_duration = effective_duration / num_segments

        base_name = self.sanitize_filename(os.path.basename(input_path))
        output_files = []

        print(
            f"\n✂️ Splitting video from {start_time:.1f}s to {start_time + effective_duration:.1f}s"
        )
        print(f"   Creating {num_segments} segments of {segment_duration:.1f}s each")

        for i in range(num_segments):
            segment_start = start_time + (i * segment_duration)
            output_path = f"{os.path.dirname(input_path)}/{base_name}_seg{i+1:02d}.mp4"

            # Use ffmpeg to extract segment
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(segment_start),  # Seek before input
                    "-i",
                    input_path,
                    "-t",
                    str(segment_duration),
                    "-c",
                    "copy",  # Copy codecs for faster processing
                    output_path,
                ],
                capture_output=True,
            )

            output_files.append(output_path)
            print(
                f"  ✓ Created segment {i+1}/{num_segments} ({segment_start:.1f}s to {segment_start + segment_duration:.1f}s)"
            )

        return output_files

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
        start_time: float = Input(
            description="Start time in seconds for video processing",
            default=0.0,
        ),
        end_time: float = Input(
            description="End time in seconds for video processing. Leave empty to process until start_time + duration",
            default=None,
        ),
        duration: float = Input(
            description="Duration in seconds to process. Ignored if end_time is set.",
            default=30.0,
        ),
        num_segments: int = Input(
            description="Number of segments to split the video into.",
            default=4,
        ),
        custom_caption: str = Input(
            description="Your custom caption for the video. Required if autocaption=False.",
            default=None,
        ),
        autocaption: bool = Input(
            description="Let AI generate a caption for your video. If False, you must provide custom_caption.",
            default=True,
        ),
        caption_prompt: str = Input(
            description="Custom prompt for the AI captioning model. Leave empty for default prompt.",
            default="Describe this video clip briefly, focusing on the main action and visual elements.",
        ),
        trigger_word: str = Input(
            description="Trigger word to include in captions (e.g., TOK, STYLE3D). Will be added at start of caption.",
            default="TOK",
        ),
        autocaption_prefix: str = Input(
            description="Text to add BEFORE caption. Example: 'a video of TOK, '",
            default=None,
        ),
        autocaption_suffix: str = Input(
            description="Text to add AFTER caption. Example: ' in the style of TOK'",
            default=None,
        ),
    ) -> Path:
        """Process a video, split it into segments, and create a training-ready zip file with captions."""

        # Input validation
        if video_url and video_file:
            print(
                "\n⚠️ Warning: Both URL and file provided. Using URL and ignoring file."
            )
        elif not video_url and not video_file:
            raise ValueError("Must provide either video_url or video_file")

        # Create working directory
        temp_dir = "/tmp/video_processing"
        videos_dir = os.path.join(temp_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        try:
            # Handle video input
            if video_url:
                print(f"\n📥 Downloading video from: {video_url}")
                filename = self.download_video(video_url, videos_dir)
                video_path = os.path.join(videos_dir, filename)
            else:
                print(f"\n📥 Processing uploaded video: {video_file.name}")
                sanitized_name = f"{self.sanitize_filename(video_file.name)}.mp4"
                video_path = os.path.join(videos_dir, sanitized_name)
                shutil.copy(str(video_file), video_path)
                filename = sanitized_name

            # Calculate effective duration
            if end_time is not None:
                if end_time <= start_time:
                    raise ValueError("end_time must be greater than start_time")
                effective_duration = end_time - start_time
            else:
                effective_duration = duration

            # Split video into segments
            segment_paths = self.split_video(
                video_path,
                start_time=start_time,
                duration=effective_duration,
                num_segments=num_segments,
            )

            # Process each segment
            for i, segment_path in enumerate(segment_paths, 1):
                base_name = os.path.splitext(os.path.basename(segment_path))[0]
                caption_path = os.path.join(videos_dir, f"{base_name}.txt")

                # Build caption components
                prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                trigger = f"{trigger_word.strip()} " if trigger_word else ""

                print(f"\n🎬 Processing segment {i}/{len(segment_paths)}")

                if not autocaption and custom_caption is None:
                    raise ValueError(
                        "When autocaption=False, you must provide a custom_caption"
                    )

                if not autocaption:
                    print("✍️ Using your custom caption")
                    final_caption = f"{prefix}{trigger}{custom_caption.strip()} (part {i}/{len(segment_paths)}){suffix}"
                else:
                    print("🤖 Generating caption using AI...")
                    final_caption = self.generate_caption(
                        segment_path,
                        prompt=caption_prompt,
                        trigger_word=trigger_word,
                        prefix=autocaption_prefix,
                        suffix=autocaption_suffix,
                    )

                # Save caption
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(final_caption.strip() + "\n")

                print(f"\n📝 Caption for segment {i}:")
                print("--------------------")
                print(final_caption)
                print("--------------------")

            # Create zip file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"processed_videos_{timestamp}.zip"

            with ZipFile(output_path, "w") as zipf:
                print(f"\n📦 Creating zip file...")
                for segment_path in segment_paths:
                    base_name = os.path.splitext(os.path.basename(segment_path))[0]
                    zipf.write(segment_path, f"videos/{os.path.basename(segment_path)}")
                    caption_path = os.path.join(videos_dir, f"{base_name}.txt")
                    zipf.write(caption_path, f"videos/{base_name}.txt")

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

            print(f"\n✨ Success! Output saved to: {output_path}")
            return Path(output_path)

        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
