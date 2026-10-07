"""
SSRF Protection Validator for Custom Base URLs.
Blocks private, loopback, and link-local IP addresses unless local development override is explicitly enabled.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from ...shared.errors import ValidationError


def is_local_dev_allowed() -> bool:
    """Check if local addresses are explicitly allowed for dev/testing."""
    return os.environ.get("ORCHESTRATION_ALLOW_LOCAL_URLS", "false").lower() in ("true", "1", "yes")


def validate_base_url(url_str: str, allow_local: bool | None = None) -> str:
    """
    Validates a custom provider base URL against SSRF threats.
    Rejects:
    - Non http/https schemes
    - Private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    - Loopback (127.0.0.0/8, ::1)
    - Link-local / AWS metadata (169.254.0.0/16, fe80::/10)
    - Carrier-grade NAT (100.64.0.0/10)
    - Multicast
    """
    if not url_str:
        return ""

    if allow_local is None:
        allow_local = is_local_dev_allowed()

    parsed = urlparse(url_str.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            f"Invalid scheme '{parsed.scheme}'. Base URL must use http or https.",
            invalid_params=[{"name": "base_url", "reason": "scheme must be http or https"}]
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValidationError(
            "Missing hostname in base URL",
            invalid_params=[{"name": "base_url", "reason": "missing hostname"}]
        )

    # If allow_local is true (e.g. local Ollama during dev), skip IP blocking
    if allow_local:
        return url_str.strip().rstrip("/")

    # Check hostname directly or resolve DNS
    try:
        # Check if direct IP
        ip = ipaddress.ip_address(hostname)
        ips = [ip]
    except ValueError:
        # Hostname: resolve IP addresses
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            ips = [ipaddress.ip_address(addr[4][0]) for addr in addr_info]
        except socket.gaierror as e:
            raise ValidationError(
                f"Failed to resolve host '{hostname}': {e}",
                invalid_params=[{"name": "base_url", "reason": f"DNS resolution failed: {e}"}]
            )

    for ip in ips:
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValidationError(
                f"Base URL host '{hostname}' resolves to blocked private or local network ({ip}). "
                "Private and loopback addresses are blocked to prevent SSRF.",
                invalid_params=[{"name": "base_url", "reason": f"blocked IP address {ip}"}]
            )

    return url_str.strip().rstrip("/")
