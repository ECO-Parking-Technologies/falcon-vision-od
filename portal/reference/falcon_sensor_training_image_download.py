"""
Falcon Vision Sensor Training Image Download Tool

Downloads training images from sensors via Cloudflare Access URLs.
No SSH/fabric required - uses direct HTTP access to storage paths.
"""

import argparse
import getpass
import json
import logging
import os
import re
import signal
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin

import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich import box
from rich.panel import Panel

# Add inference/dnn directory to path to import cloudflare_auth
inference_dnn_path = Path(__file__).parent.parent.parent / 'inference' / 'dnn'
sys.path.insert(0, str(inference_dnn_path))
from cloudflare_auth import CloudflareAccessClient

# Initialize rich console
console = Console()

# Setup logging
def setup_logging(log_file='training_image_download.log'):
    """Setup logging to file only"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    # Set requests library to warning level (too verbose at debug)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


class TrainingImage:
    """Represents a training image to be downloaded"""
    def __init__(self, url: str, filename: str, timestamp: str, size: int, camera_id: int):
        self.url = url
        self.filename = filename
        self.timestamp = timestamp  # YYYYMMDDHHmmss format (from dateTime field)
        self.size = size
        self.camera_id = camera_id

    def __repr__(self):
        return f"TrainingImage({self.filename}, {self.timestamp}, {self.size} bytes)"


class FalconVisionSensor:
    """Handles downloading training images from a single sensor"""

    def __init__(self, gateway_url: str, sensor_hostname: str, garage_name: str,
                 start_time: str, end_time: str, interval: int, cameras: int,
                 cf_client: CloudflareAccessClient, dest_path: Path, cache_path: Path = None):
        """
        Initialize sensor downloader.

        Args:
            gateway_url: Gateway URL (e.g., "https://legacy.fvg-yaamava-north.private.ecofalcondata.com")
            sensor_hostname: Sensor hostname (e.g., "FV6ef1ac")
            garage_name: Garage name for organizing downloads
            start_time: Start time in YYYYMMDDHHmm format
            end_time: End time in YYYYMMDDHHmm format
            interval: Minutes between images (0 = all images)
            cameras: Camera filter (0 = both, 1 = camera-1, 2 = camera-2)
            cf_client: Cloudflare Access client
            dest_path: Destination path for downloads
            cache_path: Path to cache directory for discovery results
        """
        self.gateway_url = gateway_url
        self.sensor_hostname = sensor_hostname
        self.garage_name = garage_name
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval
        self.cameras = cameras
        self.cf_client = cf_client
        self.dest_path = dest_path
        self.cache_path = cache_path

        # Construct sensor URL from gateway
        # Input: "https://legacy.fvg-yaamava-north.private.ecofalcondata.com"
        # Output: "https://fv6ef1ac.fvg-yaamava-north.private.ecofalcondata.com"
        parsed = urlparse(gateway_url)
        gateway_host = parsed.hostname
        domain_parts = gateway_host.split('.', 1)
        if len(domain_parts) > 1:
            garage_domain = domain_parts[1]
        else:
            garage_domain = gateway_host

        self.sensor_url = f"https://{sensor_hostname.lower()}.{garage_domain}"

        # Setup cache file path if caching is enabled
        if self.cache_path:
            self.cache_file = self.cache_path / f"{sensor_hostname}_discovery_cache.json"
        else:
            self.cache_file = None

    def load_cache(self) -> Dict:
        """Load discovery cache from JSON file"""
        if not self.cache_file or not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_cache(self, cache: Dict):
        """Save discovery cache to JSON file"""
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            pass  # Silently fail on cache write errors

    def get_training_image_files(self, year: int, month: int, day: int, hour: int, logger=None, cache=None) -> List[TrainingImage]:
        """
        Get training image files for a specific hour using the API endpoint.

        API: /training-image/files/{yyyy}/{m}/{d}/{h}
        Returns JSON: {"data": [{"cameraId": 1, "dateTime": "YYYYMMDDHHmmss", "size": bytes, "fileName": "..."}]}

        Args:
            year: Year (e.g., 2026)
            month: Month (1-12, no leading zero)
            day: Day (1-31, no leading zero)
            hour: Hour (0-23, no leading zero)
            logger: Optional logger instance
            cache: Optional cache dict to check before querying API

        Returns:
            List of TrainingImage objects
        """
        # Check cache first
        cache_key = f"{year}/{month}/{day}/{hour}"
        if cache and cache_key in cache:
            if logger:
                logger.debug(f"[{self.sensor_hostname}] Cache hit: {cache_key}")
            # Reconstruct TrainingImage objects from cached data
            images = []
            for file_info in cache[cache_key]:
                image_url = f"{self.sensor_url}/training-image/image/{file_info['fileName']}"

                # Parse timestamp - handle both ISO 8601 and YYYYMMDDHHmmss formats
                timestamp_str = file_info['dateTime']
                try:
                    if 'T' in timestamp_str:
                        dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                        timestamp_normalized = dt.strftime('%Y%m%d%H%M%S')
                    else:
                        timestamp_normalized = timestamp_str
                except ValueError:
                    timestamp_normalized = timestamp_str

                images.append(TrainingImage(
                    url=image_url,
                    filename=file_info['fileName'],
                    timestamp=timestamp_normalized,
                    size=file_info['size'],
                    camera_id=file_info['cameraId']
                ))
            return images

        # Construct API endpoint URL
        endpoint = f"{self.sensor_url}/training-image/files/{year}/{month}/{day}/{hour}"

        try:
            if logger:
                logger.debug(f"[{self.sensor_hostname}] Querying: {year}/{month}/{day}/{hour}")

            response = self.cf_client.session.get(endpoint)
            response.raise_for_status()

            data = response.json()
            images = []

            if 'data' in data:
                # Cache the raw response data
                if cache is not None:
                    cache[cache_key] = data['data']

                for file_info in data['data']:
                    # Construct image download URL
                    # API: /training-image/image/{filename}
                    image_url = f"{self.sensor_url}/training-image/image/{file_info['fileName']}"

                    # Parse timestamp - API returns ISO 8601 format (2026-01-15T21:56:57Z)
                    # Convert to YYYYMMDDHHmmss format for consistency
                    timestamp_str = file_info['dateTime']
                    try:
                        # Try ISO 8601 format first (what API actually returns)
                        if 'T' in timestamp_str:
                            dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                            timestamp_normalized = dt.strftime('%Y%m%d%H%M%S')
                        else:
                            # Already in YYYYMMDDHHmmss format
                            timestamp_normalized = timestamp_str
                    except ValueError:
                        # Fallback: use as-is
                        timestamp_normalized = timestamp_str

                    images.append(TrainingImage(
                        url=image_url,
                        filename=file_info['fileName'],
                        timestamp=timestamp_normalized,
                        size=file_info['size'],
                        camera_id=file_info['cameraId']
                    ))

                if logger and images:
                    logger.debug(f"[{self.sensor_hostname}] Found {len(images)} images at {year}/{month}/{day}/{hour}")

            return images

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # No images for this hour (normal) - cache empty result
                if cache is not None:
                    cache[cache_key] = []
                return []
            else:
                if logger:
                    logger.warning(f"[{self.sensor_hostname}] HTTP {e.response.status_code} for {endpoint}")
                return []
        except Exception as e:
            if logger:
                logger.error(f"[{self.sensor_hostname}] Error accessing {endpoint}: {e}")
            return []

    def generate_hour_ranges(self, start_dt: datetime, end_dt: datetime) -> List[tuple]:
        """
        Generate (year, month, day, hour) tuples for time range.

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            List of (year, month, day, hour) tuples
        """
        hours = []
        current = start_dt.replace(minute=0, second=0, microsecond=0)

        while current <= end_dt:
            hours.append((current.year, current.month, current.day, current.hour))
            current += timedelta(hours=1)

        return hours

    def filter_by_interval(self, images: List[TrainingImage]) -> List[TrainingImage]:
        """
        Filter images based on interval (minutes between images).

        Args:
            images: List of TrainingImage objects

        Returns:
            Filtered list of TrainingImage objects
        """
        if not images or self.interval == 0:
            return images

        filtered = []
        next_timestamp = datetime.min

        for image in sorted(images, key=lambda x: x.timestamp):
            img_dt = datetime.strptime(image.timestamp, '%Y%m%d%H%M%S')
            if img_dt >= next_timestamp:
                filtered.append(image)
                next_timestamp = img_dt + timedelta(minutes=self.interval)

        return filtered

    def download(self, verbose=True, logger=None, progress_callback=None):
        """
        Download training images from sensor.

        Args:
            verbose: If True, print progress messages. If False, only return stats.
            logger: Optional logger instance
            progress_callback: Optional callback function(sensor, status, progress_data)

        Returns:
            Dictionary with download statistics
        """
        if logger:
            logger.info(f"[{self.sensor_hostname}] Starting download")

        # Load discovery cache
        cache = self.load_cache()
        cache_hits = 0
        cache_misses = 0

        # Parse time range
        start_dt = datetime.strptime(self.start_time, '%Y%m%d%H%M')
        end_dt = datetime.strptime(self.end_time, '%Y%m%d%H%M')

        # Generate hour ranges for time range
        hour_ranges = self.generate_hour_ranges(start_dt, end_dt)

        if logger:
            logger.info(f"[{self.sensor_hostname}] Searching {len(hour_ranges)} hours for images (cache: {len(cache)} entries)")

        discovered_images = []

        for i, (year, month, day, hour) in enumerate(hour_ranges):
            # Update progress BEFORE querying (for immediate UI feedback)
            if progress_callback:
                progress_callback(self.sensor_hostname, f"Searching {year}/{month}/{day:02d}/{hour:02d}", {
                    'phase': 'discovery',
                    'progress': i + 1,
                    'total': len(hour_ranges),
                    'found': len(discovered_images)
                })

            # Track cache hits/misses
            cache_key = f"{year}/{month}/{day}/{hour}"
            was_cached = cache_key in cache

            images = self.get_training_image_files(year, month, day, hour, logger=logger, cache=cache)

            if was_cached:
                cache_hits += 1
            else:
                cache_misses += 1

            # Filter by camera if specified
            if self.cameras != 0:
                images = [img for img in images if img.camera_id == self.cameras]

            discovered_images.extend(images)

        # Save updated cache
        if cache_misses > 0:
            self.save_cache(cache)
            if logger:
                logger.info(f"[{self.sensor_hostname}] Cache stats: {cache_hits} hits, {cache_misses} misses")

        if not discovered_images:
            if logger:
                logger.info(f"[{self.sensor_hostname}] No images found in time range")
            return {
                'sensor': self.sensor_hostname,
                'discovered': 0,
                'new': 0,
                'downloaded': 0,
                'errors': 0,
                'skipped': 0
            }

        if logger:
            logger.info(f"[{self.sensor_hostname}] Discovered {len(discovered_images)} images")

        # Filter out already downloaded
        new_images = [img for img in discovered_images
                      if not (self.dest_path / img.filename).exists()]

        if not new_images:
            if logger:
                logger.info(f"[{self.sensor_hostname}] All {len(discovered_images)} images already present")
            return {
                'sensor': self.sensor_hostname,
                'discovered': len(discovered_images),
                'new': 0,
                'downloaded': 0,
                'errors': 0,
                'skipped': len(discovered_images)
            }

        if logger:
            logger.info(f"[{self.sensor_hostname}] Need to download {len(new_images)} new images")

        # Apply interval filtering
        if self.interval > 0:
            new_images = self.filter_by_interval(new_images)
            if logger:
                logger.info(f"[{self.sensor_hostname}] After interval filtering: {len(new_images)} images")

        # Create destination directory
        self.dest_path.mkdir(parents=True, exist_ok=True)

        # Download images (no progress bar in worker threads)
        success_count = 0
        error_count = 0
        error_messages = []

        for i, image in enumerate(new_images):
            if progress_callback:
                progress_callback(self.sensor_hostname, f"Downloading {image.filename}", {
                    'phase': 'download',
                    'progress': i + 1,
                    'total': len(new_images)
                })

            try:
                response = self.cf_client.session.get(image.url)
                response.raise_for_status()

                img_path = self.dest_path / image.filename
                with open(img_path, 'wb') as f:
                    f.write(response.content)

                success_count += 1
            except Exception as e:
                error_count += 1
                error_msg = f"{image.filename}: {e}"
                error_messages.append(error_msg)
                if logger:
                    logger.error(f"[{self.sensor_hostname}] Download error: {error_msg}")

        if logger:
            logger.info(f"[{self.sensor_hostname}] Complete: {success_count} downloaded, {error_count} errors")

        return {
            'sensor': self.sensor_hostname,
            'discovered': len(discovered_images),
            'new': len(new_images),
            'downloaded': success_count,
            'errors': error_count,
            'skipped': len(discovered_images) - len(new_images),
            'error_messages': error_messages
        }


def get_sensor_list(gateway_url: str, cf_client: CloudflareAccessClient) -> List[str]:
    """
    Enumerate sensors at a gateway.

    Args:
        gateway_url: Gateway URL
        cf_client: Cloudflare Access client

    Returns:
        List of sensor hostnames
    """
    endpoint = f"{gateway_url}/plugin/registered-sensor/enum"

    try:
        response = cf_client.post(endpoint, data={})
        sensor_list = response['response']['data']['sensorList']
        hostnames = [sensor['hstNm'] for sensor in sensor_list]
        return hostnames
    except Exception as e:
        print(f"Error enumerating sensors: {e}")
        return []


def get_credentials() -> tuple[str, str]:
    """
    Get Cloudflare Access credentials via interactive prompt.

    Returns:
        Tuple of (client_id, client_secret)
    """
    console.print()
    console.print(Panel(
        "[cyan]Cloudflare Access authentication required[/cyan]\n"
        "Service token credentials needed to access sensors",
        title="[bold]Authentication[/bold]",
        border_style="cyan"
    ))
    console.print()

    client_id = console.input("[cyan]CF-Access-Client-Id:[/cyan] ").strip()
    client_secret = getpass.getpass("CF-Access-Client-Secret: ")

    if not client_id or not client_secret:
        console.print("[red]Error: Credentials are required[/red]")
        raise ValueError("Cloudflare Access credentials are required")

    return client_id, client_secret


def main():
    parser = argparse.ArgumentParser(
        prog='Falcon Vision Training Image Downloader',
        description='Downloads training images from sensors via Cloudflare Access'
    )

    parser.add_argument('--yaml_config_file', type=str, required=True,
                        help='YAML config file with download parameters')
    parser.add_argument('--client-id', type=str,
                        help='CF-Access-Client-Id (overrides environment variable)')
    parser.add_argument('--client-secret', type=str,
                        help='CF-Access-Client-Secret (overrides environment variable)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Maximum number of parallel sensor downloads (default: 10)')
    parser.add_argument('--log-file', type=str, default='training_image_download.log',
                        help='Log file path (default: training_image_download.log)')
    parser.add_argument('--cache-dir', type=str, default='.training_image_cache',
                        help='Directory for discovery cache files (default: .training_image_cache)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable discovery caching (always query API)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("=" * 80)
    logger.info("Training Image Download Tool Starting")
    logger.info("=" * 80)

    # Load YAML config
    if not os.path.exists(args.yaml_config_file):
        logger.error(f"Config file not found: {args.yaml_config_file}")
        print(f"Error: Config file not found: {args.yaml_config_file}")
        sys.exit(1)

    logger.info(f"Loading config from: {args.yaml_config_file}")
    with open(args.yaml_config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get Cloudflare credentials
    if args.client_id and args.client_secret:
        client_id = args.client_id
        client_secret = args.client_secret
        logger.info("Using Cloudflare credentials from command line arguments")
    else:
        logger.info("Prompting for Cloudflare credentials")
        client_id, client_secret = get_credentials()

    # Initialize Cloudflare client
    logger.info("Initializing Cloudflare Access client")
    cf_client = CloudflareAccessClient(client_id, client_secret)

    # Get output path - default to artifacts directory next to this script
    if 'output_path' in config['download']:
        output_path = Path(config['download']['output_path'])
    else:
        # Default: artifacts directory in same dir as this script
        script_dir = Path(__file__).parent
        output_path = script_dir / 'artifacts'

    logger.info(f"Output path: {output_path}")

    # Setup cache path
    if args.no_cache:
        cache_path = None
        logger.info("Discovery caching disabled")
    else:
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Discovery cache: {cache_path}")

    # Process each garage
    all_sensors = []

    for garage_name, garage_config in config['garages'].items():
        gateway_url = garage_config['gateway_url']
        logger.info(f"Processing garage: {garage_name} ({gateway_url})")

        console.print()
        console.print(Panel(
            f"[cyan]Processing:[/cyan] {garage_config.get('display_name', garage_name)}\n"
            f"[dim]Gateway:[/dim] {gateway_url}",
            title="[bold]Garage[/bold]",
            border_style="cyan"
        ))

        # Get sensor list
        if garage_config.get('sensors') and len(garage_config['sensors']) > 0:
            # Check for wildcard '*'
            if '*' in garage_config['sensors']:
                logger.info("Wildcard '*' detected - enumerating all sensors from gateway...")
                console.print(f"[cyan]→[/cyan] Wildcard '*' detected - enumerating all sensors...")
                sensor_list = get_sensor_list(gateway_url, cf_client)
                logger.info(f"Found {len(sensor_list)} sensors (wildcard)")
                console.print(f"[cyan]→[/cyan] Found [bold]{len(sensor_list)}[/bold] sensors (wildcard)")
            else:
                sensor_list = garage_config['sensors']
                logger.info(f"Using configured sensor list: {len(sensor_list)} sensors")
                console.print(f"[cyan]→[/cyan] Using configured sensor list: [bold]{len(sensor_list)}[/bold] sensors")
        else:
            logger.info("Enumerating sensors from gateway...")
            console.print(f"[cyan]→[/cyan] Enumerating sensors...")
            sensor_list = get_sensor_list(gateway_url, cf_client)
            logger.info(f"Found {len(sensor_list)} sensors")
            console.print(f"[cyan]→[/cyan] Found [bold]{len(sensor_list)}[/bold] sensors")

        if not sensor_list:
            logger.warning(f"No sensors found for {garage_name}")
            console.print(f"[yellow]Warning: No sensors found for {garage_name}[/yellow]")
            continue

        # Create sensor downloader objects
        for sensor_hostname in sensor_list:
            # Directory structure: {output_path}/{garage_name}/training_images/{sensor_hostname}
            sensor_dest = output_path / garage_name / 'training_images' / sensor_hostname

            sensor = FalconVisionSensor(
                gateway_url=gateway_url,
                sensor_hostname=sensor_hostname,
                garage_name=garage_name,
                start_time=garage_config['start-time'],
                end_time=garage_config['end-time'],
                interval=garage_config['interval'],
                cameras=garage_config['cameras'],
                cf_client=cf_client,
                dest_path=sensor_dest,
                cache_path=cache_path
            )
            all_sensors.append(sensor)

        logger.info(f"Created {len(all_sensors)} sensor downloaders")

    # Download from all sensors using multithreading with rich display
    logger.info("="*80)
    logger.info(f"Starting parallel downloads: {len(all_sensors)} sensors, {args.max_workers} workers")
    logger.info("="*80)

    console.print()

    # Format interval display
    interval = garage_config['interval']
    if interval == 0:
        interval_display = "All images (no filtering)"
    elif interval == 60:
        interval_display = f"{interval} minutes (hourly sampling)"
    elif interval >= 60:
        hours = interval // 60
        interval_display = f"{interval} minutes ({hours}-hour sampling)"
    else:
        interval_display = f"{interval} minutes"

    console.print(Panel(
        f"[cyan]Starting parallel downloads[/cyan]\n"
        f"Garage: {garage_config.get('display_name', garage_name)}\n"
        f"Sensors: {len(all_sensors)}\n"
        f"Workers: {args.max_workers}\n"
        f"Time range: {garage_config['start-time']} → {garage_config['end-time']}\n"
        f"Interval: {interval_display}",
        title="[bold]Download Configuration[/bold]",
        border_style="cyan"
    ))
    console.print()

    completed = 0
    total_discovered = 0
    total_downloaded = 0
    total_errors = 0
    total_skipped = 0

    # Track sensor statuses for live display
    sensor_statuses = {}
    worker_progress = {}  # Track what each worker is doing

    def progress_callback(sensor_name, status_msg, progress_data):
        """Callback to update worker progress"""
        worker_progress[sensor_name] = {
            'status': status_msg,
            'phase': progress_data.get('phase', 'unknown'),
            'progress': progress_data.get('progress', 0),
            'total': progress_data.get('total', 0)
        }

    def download_sensor(sensor):
        """Download images from a single sensor (worker thread)"""
        try:
            # download() now returns stats instead of printing
            stats = sensor.download(verbose=False, logger=logger, progress_callback=progress_callback)
            return stats
        except Exception as err:
            # Only log errors in worker threads
            logger.error(f"[{sensor.sensor_hostname}] Exception in download: {err}", exc_info=True)
            return {
                'sensor': sensor.sensor_hostname,
                'discovered': 0,
                'new': 0,
                'downloaded': 0,
                'errors': 1,
                'skipped': 0,
                'error_messages': [f"Exception: {err}"]
            }

    def generate_status_table():
        """Generate a rich table showing current download status"""
        # Active workers table
        if worker_progress:
            worker_table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow", title="[bold yellow]Active Workers[/bold yellow]")
            worker_table.add_column("Sensor", style="yellow", width=12)
            worker_table.add_column("Phase", width=15)
            worker_table.add_column("Status", width=50)
            worker_table.add_column("Progress", justify="right", width=12)

            for sensor_name, progress in list(worker_progress.items())[:args.max_workers]:
                phase = progress['phase']
                status = progress['status']
                prog = progress['progress']
                total = progress['total']

                phase_display = {
                    'discovery': '🔍 Searching',
                    'download': '⬇️  Downloading'
                }.get(phase, phase)

                progress_display = f"{prog}/{total}" if total > 0 else "-"

                worker_table.add_row(
                    sensor_name,
                    phase_display,
                    status[:50],  # Truncate long status
                    progress_display
                )

        # Completed sensors table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", title="[bold cyan]Completed Sensors (last 10)[/bold cyan]")
        table.add_column("Sensor", style="cyan", width=12)
        table.add_column("Status", width=20)
        table.add_column("Discovered", justify="right", width=10)
        table.add_column("Downloaded", justify="right", width=10, style="green")
        table.add_column("Skipped", justify="right", width=10, style="yellow")
        table.add_column("Errors", justify="right", width=8, style="red")

        # Show only last 10 completed sensors
        recent_sensors = list(sensor_statuses.items())[-10:]
        for sensor_name, stats in recent_sensors:
            status_text = stats['status']
            status_color = stats['color']

            table.add_row(
                sensor_name,
                f"[{status_color}]{status_text}[/{status_color}]",
                str(stats['discovered']),
                str(stats['downloaded']),
                str(stats['skipped']),
                str(stats['errors']) if stats['errors'] > 0 else "-"
            )

        # Add summary row
        table.add_section()
        error_display = f"[bold red]{total_errors}[/bold red]" if total_errors > 0 else "[dim]-[/dim]"
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[cyan]{completed}/{len(all_sensors)} completed[/cyan]",
            f"[bold]{total_discovered}[/bold]",
            f"[bold green]{total_downloaded}[/bold green]",
            f"[bold yellow]{total_skipped}[/bold yellow]",
            error_display
        )

        # Combine tables
        from rich.console import Group
        if worker_progress:
            return Group(worker_table, "", table)
        else:
            return table

    # Set up signal handler for clean Ctrl-C shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            console.print("\n[yellow]⚠️  Shutdown requested (Ctrl-C). Finishing current downloads...[/yellow]")
            console.print("[dim]   Press Ctrl-C again to force quit.[/dim]\n")
            shutdown_requested = True
        else:
            console.print("\n[red bold]⚠️  Force quit![/red bold]")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Use ThreadPoolExecutor for parallel downloads with Live display
    try:
        with Live(generate_status_table(), refresh_per_second=4, console=console) as live:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all download tasks
                future_to_sensor = {executor.submit(download_sensor, sensor): sensor
                                   for sensor in all_sensors}

                # Process completed downloads as they finish
                for future in as_completed(future_to_sensor):
                    if shutdown_requested:
                        # Cancel remaining futures on shutdown
                        for f in future_to_sensor:
                            f.cancel()
                        console.print("\n[yellow]Cancelling remaining downloads...[/yellow]")
                        break

                    stats = future.result()
                    completed += 1

                    # Accumulate totals
                    total_discovered += stats['discovered']
                    total_downloaded += stats['downloaded']
                    total_errors += stats['errors']
                    total_skipped += stats['skipped']

                    # Determine status and color
                    sensor_name = stats['sensor']
                    if stats['errors'] > 0:
                        status = f"✗ {stats['errors']} errors"
                        color = "red"
                        logger.warning(f"[{sensor_name}] Completed with {stats['errors']} errors")
                    elif stats['discovered'] == 0:
                        status = "- No images"
                        color = "dim"
                        logger.info(f"[{sensor_name}] No images found")
                    elif stats['downloaded'] == 0:
                        status = "✓ Already complete"
                        color = "yellow"
                        logger.info(f"[{sensor_name}] All images already present")
                    else:
                        status = "✓ Complete"
                        color = "green"
                        logger.info(f"[{sensor_name}] Successfully downloaded {stats['downloaded']} images")

                    # Store sensor status
                    sensor_statuses[sensor_name] = {
                        'status': status,
                        'color': color,
                        'discovered': stats['discovered'],
                        'downloaded': stats['downloaded'],
                        'skipped': stats['skipped'],
                        'errors': stats['errors']
                    }

                    # Remove from worker progress (sensor is complete)
                    if sensor_name in worker_progress:
                        del worker_progress[sensor_name]

                    # Update live display
                    live.update(generate_status_table())

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl-C)")
        console.print("\n[yellow]⚠️  Interrupted! Cleaning up...[/yellow]")
        pass

    # Build summary message
    logger.info("="*80)
    logger.info("Download Complete!")
    logger.info(f"Total Sensors: {len(all_sensors)}")
    logger.info(f"Images Discovered: {total_discovered:,}")
    logger.info(f"Images Downloaded: {total_downloaded:,}")
    logger.info(f"Already Present: {total_skipped:,}")
    logger.info(f"Download Errors: {total_errors}")
    logger.info("="*80)

    error_color = "red" if total_errors > 0 else "green"
    summary_text = (
        f"[bold green]Download Complete![/bold green]\n\n"
        f"Total Sensors:     [cyan]{len(all_sensors)}[/cyan]\n"
        f"Images Discovered: [cyan]{total_discovered:,}[/cyan]\n"
        f"Images Downloaded: [bold green]{total_downloaded:,}[/bold green]\n"
        f"Already Present:   [yellow]{total_skipped:,}[/yellow]\n"
        f"Download Errors:   [{error_color}]{total_errors}[/{error_color}]"
    )

    console.print()
    console.print(Panel(
        summary_text,
        title="[bold]Summary[/bold]",
        border_style="green"
    ))

    console.print(f"\n[dim]Log file: {args.log_file}[/dim]")


if __name__ == "__main__":
    main()
