#!/usr/bin/env python3
"""Download model weights from RunPod to local filesystem.

Usage:
    python scripts/download_from_runpod.py --help

Requirements:
    - SSH access to RunPod instance
    - rsync installed on local machine
    - SSH key configured for RunPod access
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download model weights/checkpoints from RunPod to local filesystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all checkpoints from default output dir
  python scripts/download_from_runpod.py user@runpod-instance

  # Download specific checkpoint
  python scripts/download_from_runpod.py user@runpod-instance \\
      --source checkpoints/checkpoint-500 \\
      --dest ./my_checkpoints

  # Download with custom SSH key and port
  python scripts/download_from_runpod.py user@runpod-instance \\
      --ssh-key ~/.ssh/runpod_key \\
      --port 2222

  # Download final model only (not checkpoints)
  python scripts/download_from_runpod.py user@runpod-instance \\
      --final-only
        """
    )
    parser.add_argument(
        "host",
        type=str,
        help="RunPod SSH host (e.g., user@1.2.3.4 or user@runpod-instance)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="checkpoints",
        help="Source directory on RunPod (default: checkpoints)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="./checkpoints",
        help="Local destination directory (default: ./checkpoints)",
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        default=None,
        help="Path to SSH private key (default: use ssh agent or ~/.ssh/id_rsa)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22,
        help="SSH port (default: 22)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only download final model, skip intermediate checkpoints",
    )
    parser.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="Exclude checkpoint-* directories, only keep final model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Use compression during transfer (default: True)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression during transfer",
    )
    return parser.parse_args()


def build_rsync_command(args: argparse.Namespace) -> list[str]:
    """Build rsync command with appropriate options.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of command arguments for subprocess.
    """
    cmd = ["rsync", "-avz", "--progress"]

    # Add SSH options
    ssh_opts = []
    if args.ssh_key:
        ssh_opts.extend(["-i", args.ssh_key])
    if args.port != 22:
        ssh_opts.extend(["-p", str(args.port)])

    if ssh_opts:
        cmd.extend(["-e", " ".join(["ssh"] + ssh_opts)])

    # Add exclude options
    if args.final_only or args.exclude_checkpoints:
        cmd.extend(["--exclude", "checkpoint-*"])

    # Dry run
    if args.dry_run:
        cmd.append("--dry-run")

    # Compression
    if args.no_compress:
        cmd.remove("z")  # Remove -z flag

    # Source path (on remote)
    source = f"{args.host}:{args.source}/"

    # Destination path (local)
    dest = args.dest

    cmd.extend([source, dest])
    return cmd


def check_rsync_available() -> bool:
    """Check if rsync is available on the system.

    Returns:
        True if rsync is available, False otherwise.
    """
    try:
        subprocess.run(
            ["rsync", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_ssh_connection(host: str, port: int, ssh_key: str | None) -> bool:
    """Check if SSH connection to RunPod is possible.

    Args:
        host: SSH host string.
        port: SSH port.
        ssh_key: Path to SSH key (optional).

    Returns:
        True if connection succeeds, False otherwise.
    """
    cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(host)

    try:
        subprocess.run(
            cmd + ["echo", "connection_ok"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    args = parse_args()

    # Check rsync availability
    if not check_rsync_available():
        print("Error: rsync is not installed or not in PATH", file=sys.stderr)
        print("Install rsync:", file=sys.stderr)
        print("  - macOS: brew install rsync", file=sys.stderr)
        print("  - Ubuntu/Debian: sudo apt-get install rsync", file=sys.stderr)
        print("  - CentOS/RHEL: sudo yum install rsync", file=sys.stderr)
        return 1

    # Check SSH connection
    print(f"Checking SSH connection to {args.host}...")
    if not check_ssh_connection(args.host, args.port, args.ssh_key):
        print("Error: Cannot connect to RunPod via SSH", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  1. Ensure you have SSH access to the RunPod instance", file=sys.stderr)
        print("  2. Check that SSH key is added to ssh agent: ssh-add ~/.ssh/your_key", file=sys.stderr)
        print("  3. Verify host format: user@ip-address", file=sys.stderr)
        print("  4. Try manual SSH test: ssh -i <key> -p <port> user@host", file=sys.stderr)
        return 1
    print("SSH connection OK!")

    # Build rsync command
    cmd = build_rsync_command(args)

    # Create destination directory if needed
    Path(args.dest).mkdir(parents=True, exist_ok=True)

    # Print what we're doing
    print(f"\nDownloading from: {args.host}:{args.source}/")
    print(f"              to: {args.dest}/")
    if args.dry_run:
        print("(DRY RUN - no files will be downloaded)")

    # Execute rsync
    print(f"\nRunning: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nDownload interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error during download: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
