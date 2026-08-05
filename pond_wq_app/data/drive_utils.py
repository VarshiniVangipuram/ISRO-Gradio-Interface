"""
data/drive_utils.py
===================
Google Drive link downloading, filename resolution, and result CSV path helpers.
"""

import os
import re
import mimetypes
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import requests

from ..config import GENERATED_RESULTS_FOLDER, _DRIVE_URL_FILENAME_CACHE


def _filename_from_content_disposition(header_value):
    """Extract a filename from an HTTP Content-Disposition header."""
    if not header_value:
        return None

    utf8_match = re.search(
        r"filename\*\s*=\s*UTF-8''([^;]+)",
        str(header_value),
        flags=re.I,
    )
    if utf8_match:
        return os.path.basename(unquote(utf8_match.group(1).strip().strip('"')))

    plain_match = re.search(
        r'filename\s*=\s*"([^"]+)"',
        str(header_value),
        flags=re.I,
    )
    if plain_match:
        return os.path.basename(plain_match.group(1).strip())

    plain_match = re.search(
        r"filename\s*=\s*([^;]+)",
        str(header_value),
        flags=re.I,
    )
    if plain_match:
        return os.path.basename(plain_match.group(1).strip().strip('"'))

    return None


def _google_drive_file_id(url):
    """Read a Google Drive file ID from common sharing-link formats."""
    url = str(url).strip()

    path_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if path_match:
        return path_match.group(1)

    query = parse_qs(urlparse(url).query)
    if query.get("id"):
        return query["id"][0]

    return None


def _safe_source_filename(upload_file=None, drive_path=None, fallback="dataset.csv"):
    """Resolve a useful source filename for uploaded files, Drive paths and URLs."""
    if upload_file is not None:
        candidate = getattr(upload_file, "orig_name", None)
        if not candidate:
            candidate = getattr(upload_file, "name", None)
        if not candidate:
            candidate = str(upload_file)
        candidate = os.path.basename(str(candidate))
        if candidate:
            return candidate

    if drive_path and str(drive_path).strip():
        path = str(drive_path).strip()
        if path.startswith(("http://", "https://")):
            cached = _DRIVE_URL_FILENAME_CACHE.get(path)
            if cached:
                return cached
        else:
            candidate = os.path.basename(path.rstrip("/"))
            if candidate:
                return candidate

    return fallback


def _result_csv_path(source_filename, suffix="predictions"):
    """Create a clean runtime CSV filename derived from the input dataset."""
    source_filename = os.path.basename(str(source_filename or "dataset.csv"))
    source_stem = Path(source_filename).stem
    source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_stem).strip("_.")
    source_stem = source_stem or "dataset"

    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(suffix)).strip("_.")
    filename = f"{source_stem}_{suffix}.csv" if suffix else f"{source_stem}.csv"

    GENERATED_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    return str(GENERATED_RESULTS_FOLDER / filename)


def download_drive_link(url):
    """
    Download a trusted Google Drive sharing link into a named BytesIO object.

    The returned object carries the original filename whenever Google Drive
    provides it. The filename is cached so result downloads can use the same
    dataset name instead of a generic operational_results.csv filename.
    """
    url = str(url).strip()
    file_id = _google_drive_file_id(url)
    if not file_id:
        raise ValueError(
            "Invalid Google Drive link. Use a sharing link containing '/d/FILE_ID' "
            "or an '?id=FILE_ID' parameter."
        )

    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(
            direct_link,
            allow_redirects=True,
            timeout=120,
        )
        response.raise_for_status()
    except Exception as exc:
        raise ValueError(
            "Failed to download the Google Drive file. Ensure sharing is set to "
            f"'Anyone with the link can view'. Details: {exc}"
        ) from exc

    filename = _filename_from_content_disposition(
        response.headers.get("Content-Disposition")
    )

    if not filename:
        final_path_name = os.path.basename(urlparse(response.url).path)
        if final_path_name and "." in final_path_name:
            filename = unquote(final_path_name)

    if not filename:
        content_type = str(response.headers.get("Content-Type", "")).split(";")[0].strip()
        extension = mimetypes.guess_extension(content_type) or ".csv"
        if extension == ".jpe":
            extension = ".jpg"
        filename = f"google_drive_{file_id}{extension}"

    filename = os.path.basename(filename)
    _DRIVE_URL_FILENAME_CACHE[url] = filename

    file_obj = BytesIO(response.content)
    file_obj.name = filename
    return file_obj
