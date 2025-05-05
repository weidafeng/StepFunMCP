import os
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz
import shutil
import subprocess
from typing import Iterator, Union
from stepfun_mcp.const import *
from stepfun_mcp.exceptions import StepFunAPIError, StepFunAuthError, StepFunRequestError


def is_file_writeable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent_dir = path.parent
    return os.access(parent_dir, os.W_OK)


def build_output_file(
    tool: str, text: str, output_path: Path, extension: str, full_id: bool = False
) -> Path:
    id = text if full_id else text[:10]

    output_file_name = f"{tool}_{id.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
    return output_path / output_file_name


def build_output_path(
    output_directory: str | None, base_path: str | None = None, is_test: bool = False
) -> Path:
    # Set default base_path to desktop if not provided
    if base_path is None:
        base_path = str(Path.home() / "Desktop")
    
    # Handle output path based on output_directory
    if output_directory is None:
        output_path = Path(os.path.expanduser(base_path))
    elif not os.path.isabs(os.path.expanduser(output_directory)):
        output_path = Path(os.path.expanduser(base_path)) / Path(output_directory)
    else:
        output_path = Path(os.path.expanduser(output_directory))

    # Safety checks and directory creation
    if is_test:
        return output_path
    if not is_file_writeable(output_path):
        raise MinimaxMcpError(f"Directory ({output_path}) is not writeable")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def find_similar_filenames(
    target_file: str, directory: Path, threshold: int = 70
) -> list[tuple[str, int]]:
    """
    Find files with names similar to the target file using fuzzy matching.

    Args:
        target_file (str): The reference filename to compare against
        directory (str): Directory to search in (defaults to current directory)
        threshold (int): Similarity threshold (0 to 100, where 100 is identical)

    Returns:
        list: List of similar filenames with their similarity scores
    """
    target_filename = os.path.basename(target_file)
    similar_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if (
                filename == target_filename
                and os.path.join(root, filename) == target_file
            ):
                continue
            similarity = fuzz.token_sort_ratio(target_filename, filename)

            if similarity >= threshold:
                file_path = Path(root) / filename
                similar_files.append((file_path, similarity))

    similar_files.sort(key=lambda x: x[1], reverse=True)

    return similar_files


def try_find_similar_files(
    filename: str, directory: Path, take_n: int = 5
) -> list[Path]:
    similar_files = find_similar_filenames(filename, directory)
    if not similar_files:
        return []

    filtered_files = []

    for path, _ in similar_files[:take_n]:
        if check_audio_file(path):
            filtered_files.append(path)

    return filtered_files


def check_audio_file(path: Path) -> bool:
    audio_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
    }
    return path.suffix.lower() in audio_extensions


def process_input_file(file_path: str, audio_content_check: bool = True) -> Path:
    if not os.path.isabs(file_path) and not os.environ.get(ENV_MINIMAX_MCP_BASE_PATH):
        raise MinimaxMcpError(
            "File path must be an absolute path if MINIMAX_MCP_BASE_PATH is not set"
        )
    path = Path(file_path)
    if not path.exists() and path.parent.exists():
        parent_directory = path.parent
        similar_files = try_find_similar_files(path.name, parent_directory)
        similar_files_formatted = ",".join([str(file) for file in similar_files])
        if similar_files:
            raise MinimaxMcpError(
                f"File ({path}) does not exist. Did you mean any of these files: {similar_files_formatted}?"
            )
        raise MinimaxMcpError(f"File ({path}) does not exist")
    elif not path.exists():
        raise MinimaxMcpError(f"File ({path}) does not exist")
    elif not path.is_file():
        raise MinimaxMcpError(f"File ({path}) is not a file")

    if audio_content_check and not check_audio_file(path):
        raise MinimaxMcpError(f"File ({path}) is not an audio or video file")
    return path


def is_installed(lib_name: str) -> bool:
    return shutil.which(lib_name) is not None


def play(
    audio: Union[bytes, Iterator[bytes]]
) -> None:
    if isinstance(audio, Iterator):
        audio = b"".join(audio)

    if not is_installed("ffplay"):
        message = (
            "ffplay from ffmpeg not found, necessary to play audio. "
            "mac: install it with 'brew install ffmpeg'. "
            "linux or windows: install it from https://ffmpeg.org/"
        )
        raise ValueError(message)
    
    args = ["ffplay", "-autoexit", "-", "-nodisp"]
    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=audio)

    proc.poll()


import base64
import hashlib
import logging
import mimetypes
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image as PILImage # Use Pillow for image processing
import io

from .const import RESOURCE_MODE_LOCAL, RESOURCE_MODE_URL

logger = logging.getLogger(__name__)

# --- Resource ID and Path Generation ---

def generate_resource_id(prefix: str = "res") -> str:
    """Generates a unique resource ID."""
    timestamp = int(time.time() * 1000)
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}-{timestamp}-{random_str}"

def get_resource_path(base_path: Path, resource_id: str, file_extension: str) -> Path:
    """Constructs the full path for saving a resource locally."""
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    return base_path / f"{resource_id}{file_extension}"

# --- File Saving ---

def save_audio_to_file(audio_bytes: bytes, file_path: Path):
    """Saves audio bytes to a local file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Audio saved successfully to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")
        raise


# --- URL Validation and Downloading ---

def validate_url(url: str) -> bool:
    """Checks if a string is a valid HTTP/HTTPS URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except ValueError:
        return False

def download_file(url: str, target_path: Path) -> None:
    """Downloads a file from a URL to a target path."""
    logger.info(f"Downloading file from {url} to {target_path}")
    try:
        response = requests.get(url, stream=True, timeout=30) # Add timeout
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"File downloaded successfully to {target_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise IOError(f"Failed to download file: {e}") from e
    except IOError as e:
        logger.error(f"Failed to save downloaded file to {target_path}: {e}")
        raise

# --- Image Processing ---

def get_image_mime_type(image_bytes: bytes) -> Optional[str]:
    """Tries to determine the MIME type of image bytes using Pillow."""
    try:
        with io.BytesIO(image_bytes) as img_buffer:
            with PILImage.open(img_buffer) as img:
                format = img.format
                if format:
                    return PILImage.MIME.get(format.upper())
    except Exception as e:
        logger.warning(f"Could not determine image MIME type: {e}")
    return None # Return None if type cannot be determined

async def image_to_base64_data_uri(url_str: str, base_path: Path, resource_mode: str) -> str:
    """
    Converts an image URL (http, https, file) or a local path string
    into a base64 data URI (data:image/...;base64,...).
    Downloads the image if it's a remote URL.
    """
    logger.debug(f"Converting image source to base64 data URI: {url_str}")
    image_bytes: Optional[bytes] = None
    mime_type: Optional[str] = None
    parsed_url = urlparse(url_str)

    try:
        if parsed_url.scheme in ["http", "https"]:
            # Download remote URL
            temp_id = generate_resource_id("temp-img")
            # Use a temporary file extension, actual type determined later
            temp_path = get_resource_path(base_path, temp_id, ".tmp")
            try:
                download_file(url_str, temp_path)
                with open(temp_path, "rb") as f:
                    image_bytes = f.read()
                # Clean up temporary file
                os.remove(temp_path)
            except (IOError, requests.exceptions.RequestException) as e:
                raise ValueError(f"Failed to download or read image from URL {url_str}: {e}") from e
            finally:
                 # Ensure temp file is removed even if reading fails
                 if temp_path.exists():
                     try:
                         os.remove(temp_path)
                     except OSError as e:
                         logger.warning(f"Could not remove temporary file {temp_path}: {e}")


        elif parsed_url.scheme == "file":
            # Handle file URI
            file_path = Path(parsed_url.path)
            if not file_path.is_file():
                raise ValueError(f"File not found at URI: {url_str}")
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            mime_type = mimetypes.guess_type(file_path)[0]

        elif parsed_url.scheme == "" and Path(url_str).expanduser().is_file():
             # Handle local path string
             file_path = Path(url_str).expanduser()
             with open(file_path, "rb") as f:
                 image_bytes = f.read()
             mime_type = mimetypes.guess_type(file_path)[0]

        elif url_str.startswith("data:image"):
             # Already a data URI, just return it (maybe validate?)
             logger.debug("Input is already a data URI.")
             # Basic validation: check for base64 marker
             if ";base64," in url_str:
                 return url_str
             else:
                 raise ValueError("Invalid data URI format: missing ';base64,'")

        else:
            raise ValueError(f"Unsupported image source format or file not found: {url_str}")

        if image_bytes is None:
             raise ValueError(f"Could not load image bytes from source: {url_str}")

        # Try to determine MIME type if not already known
        if mime_type is None:
            mime_type = get_image_mime_type(image_bytes)

        # Fallback MIME type if detection fails
        if mime_type is None:
            mime_type = "image/png" # Default fallback
            logger.warning(f"Could not determine image MIME type for {url_str}, defaulting to {mime_type}")

        # Encode to base64
        b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{b64_encoded}"
        logger.debug(f"Successfully converted {url_str} to data URI (mime: {mime_type}).")
        return data_uri

    except Exception as e:
        logger.error(f"Error converting image source '{url_str}' to base64 data URI: {e}")
        # Re-raise as ValueError to be caught by the server tool handler
        raise ValueError(f"Error processing image source '{url_str}': {e}") from e


# --- Other Utilities (Add as needed) ---

# Example: Function to calculate hash (if needed for caching or comparison)
def calculate_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Calculates the hash of byte data."""
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()



def save_image_to_file(base64_string: str, file_path: Path):
    """Saves image bytes to a local file."""
    import base64
    from PIL import Image
    from io import BytesIO

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.save(file_path)
    logger.info(f"Image saved successfully to {file_path}")