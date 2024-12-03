import whisper
import os
from pathlib import Path
import subprocess
import sys

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not found in system PATH")
        print("\nTo install ffmpeg:")
        if os.name == 'nt':  # Windows
            print("1. Using chocolatey (recommended):")
            print("   choco install ffmpeg")
            print("\n2. Or download from: https://www.gyan.dev/ffmpeg/builds/")
            print("   And add it to your system PATH")
        else:  # Unix/Linux/Mac
            print("Run: sudo apt-get install ffmpeg  # For Ubuntu/Debian")
            print("Or:  brew install ffmpeg          # For MacOS")
        return False

def transcribe_audio(file_path, output_dir=None):
    """
    Transcribe an audio file using OpenAI's Whisper model
    
    Args:
        file_path (str): Path to the audio file
        output_dir (str, optional): Directory to save the transcript. If None, saves in same directory as audio
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None

        # Load the Whisper model (will download on first run)
        print(f"Loading Whisper model...")
        model = whisper.load_model("large", device="cpu")  # Explicitly use CPU
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the filename without extension
        file_name = Path(file_path).stem
        
        print(f"Transcribing {file_name}...")
        
        # Perform the transcription
        result = model.transcribe(
            file_path,
            fp16=False  # Disable FP16 to avoid warnings on CPU
        )
        
        # Save the transcript
        output_file = os.path.join(output_dir, f"{file_name}_transcript.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"Transcription saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

def main():
    # First check if ffmpeg is installed
    if not check_ffmpeg():
        sys.exit(1)

    # Directory containing the audio files
    directory = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241101 Angus Gerro Audio"
    
    # List of files to transcribe
    files = [
        "AUDIO-2024-10-26-13-38-56 (1).m4a",
        "AUDIO-2024-10-26-13-38-56.m4a"
    ]
    
    # Verify directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            transcribe_audio(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()