# Video Dataset Creator

Turn videos into AI training data with just a few clicks!

This tool chops your videos into small clips and adds AI-generated captions - perfect for training video generation models like Hunyuan-Video, ModelScope Video, or other video generation models.

## What It Does

- ✂️ **Chops Videos**: Finds the best scene cuts automatically using PySceneDetect with multiple detection algorithms
- 🤖 **Adds Captions**: Uses Qwen2-VL vision-language model to generate detailed descriptions of each clip
- 🌟 **Quality Picker**: Selects only the good scenes based on brightness, contrast, and blur detection
- 👀 **Preview Mode**: Get actual scene video clips without captions to check before creating full dataset
- 🎬 **Handles Any Source**: Works with YouTube, local files, and direct video URLs

## Super Simple Start

```bash
# Just paste a YouTube link
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE"

# Or use your own video file
cog predict -i video_file=@your_video.mp4

# Want to get the scene clips first? (Returns MP4s, no captions yet)
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i preview_only=true
```

## The Basics (All You Need)

For most videos, you only need to provide a YouTube URL. Everything else is optional! The tool uses smart defaults to handle most cases.

```bash
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE"
```

## What's Happening Under the Hood

1. **Download & Prep**: Downloads video and converts to consistent format
2. **Scene Detection**: Finds natural cuts in the video using content analysis
3. **Quality Filtering**: Picks the best scenes based on image quality metrics
4. **AI Captioning**: Generates detailed captions for each clip using Qwen2-VL
5. **Dataset Creation**: Packages everything into a ready-to-use zip file

## Want to Cut a Specific Section?

If you only care about a part of the video (like 20-30 seconds in):

```bash
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i start_time=20 -i end_time=30
```

This will ignore everything except that 10-second window - super useful for music videos, trailers, or any video where only certain parts matter.

## Cool Things You Can Customize

### Basic Stuff

- `video_url`: YouTube link or any direct video URL
- `video_file`: Your own video file (most formats supported)
- `preview_only`: Get MP4 scene clips without captions (returns List[Path] to the video files)

### Scene Finding (Technical Details)

- `detection_mode`: Scene detection algorithm:
  - `content`: Best for hard cuts and most videos (uses content difference between frames)
  - `adaptive`: Best for camera movement and panning shots (adapts to motion)
  - `threshold`: Best for fades, dissolves and lighting changes (uses absolute thresholds)
- `min_scene_length`: Shortest clip to grab in seconds (default: 1.0)
- `max_scene_length`: Longest clip to grab in seconds (default: 5.0)
- `num_scenes`: How many scenes to keep (default: 4, use 0 for all detected scenes)

### Time Stuff

- `start_time`: Where to start in the video in seconds (for skipping intros/credits)
- `end_time`: Where to stop in seconds (0 = end of video)
- `skip_intro`: Skip the first 10 seconds (true/false) - great for YouTube videos with intros
- `target_fps`: Frame rate (24 for film look, 30 for video, -1 keeps original)

### Quality Options

- `quality`: How good the output looks:
  - `fast`: Lower quality, smaller files, great for testing (CRF 28)
  - `balanced`: Good balance of quality and size (CRF 22)
  - `high`: Best quality, larger files, great for final datasets (CRF 17)

### Caption Options

- `caption_style`: How detailed the AI-generated captions should be:
  - `minimal`: Short, concise descriptions (1-2 sentences)
  - `detailed`: Full, rich descriptions of scenes (default)
  - `custom`: Use your own custom prompt
- `custom_caption`: Your own caption prompt (used when caption_style is "custom")
- `trigger_word`: Special word for model training (default: "TOK") - appears at the start of every caption
- `autocaption_prefix`: Text to add before caption (like "a video of")
- `autocaption_suffix`: Text to add after caption (like "in cinematic style")

## Examples for Different Needs

```bash
# First extract scenes to check quality (returns the MP4 clips only)
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i preview_only=true

# Get 6 scenes with your own caption style
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i num_scenes=6 -i caption_style="custom" -i custom_caption="Describe this cinematic scene in detail"

# Skip boring intro and better handle camera movement
cog predict -i video_file=@your_video.mp4 -i skip_intro=true -i detection_mode="adaptive"

# High quality output with custom trigger word for specific model fine-tuning
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i quality="high" -i trigger_word="CINEMATIC"

# Focus on just 15-45 seconds, keep all scenes, use minimal captions
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i start_time=15 -i end_time=45 -i num_scenes=0 -i caption_style="minimal"

# Create dataset for music video training with custom caption format
cog predict -i video_url="https://www.youtube.com/watch?v=EXAMPLE" -i autocaption_prefix="a music video showing" -i autocaption_suffix="with dynamic lighting and camera movement" -i trigger_word="MUSICVIDEO"
```

## What You Get

When using normal mode:
- A zip file with video clips (.mp4), caption text files (.txt), and preview image

When using `preview_only=true`:
- Just the MP4 scene clips (returned as List[Path]) without captions

## Technical Details for AI Engineers

- **Video Processing**: Uses ffmpeg with x264 encoding for maximum quality and compatibility
- **Scene Detection**: Powered by PySceneDetect with three different detection algorithms
- **Caption Generation**: Uses Qwen2-VL (7B params) running with bfloat16 and Flash Attention 2
- **Quality Analysis**: OpenCV-based metrics for brightness, contrast, and blur detection
- **Format**: The output format is compatible with most video generation model training pipelines

## Use Cases

- Training Hunyuan-Video models
- Fine-tuning ModelScope text-to-video
- Creating custom video generation styles
- Building specialized video domain adapters
- Testing video embedding and understanding models

## Need More Control?

For advanced users, here are some pro tips:
- Use `preview_only=true` to get scene clips first, check quality, then run again for full dataset
- Combine `start_time`/`end_time` with detection modes for perfect scene extraction
- Use `min_scene_length` and `max_scene_length` to control clip duration ranges
- Customize `trigger_word` + `autocaption_prefix`/`suffix` for model-specific prompting
- Try different `detection_mode` settings for different types of content
