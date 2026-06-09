import os
import math
import subprocess

def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return float(result.stdout.strip())


def split_video(video_path, num_splits=3):
    duration = get_video_duration(video_path)
    split_duration = duration / num_splits

    output_dir = "splits"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        start_time = i * split_duration

        # Last split gets remaining duration
        if i == num_splits - 1:
            current_duration = duration - start_time
        else:
            current_duration = split_duration

        output_file = os.path.join(
            output_dir,
            f"split_{i+1}.mp4"
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-t",
            str(current_duration),
            "-c",
            "copy",
            output_file,
        ]

        subprocess.run(command, check=True)

        print(f"Created: {output_file}")

    print("Video splitting completed!")


# Example usage
split_video("data/xyz.mp4", num_splits=3)