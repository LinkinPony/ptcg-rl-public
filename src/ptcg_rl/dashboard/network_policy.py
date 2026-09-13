"""Loopback-only network policy for the training dashboard."""

from __future__ import annotations

import ipaddress

_TEST_HOSTS = frozenset({"testclient", "testserver"})


def is_loopback_host(value: str | None, *, allow_test_hosts: bool = False) -> bool:
    """Return whether a host is an explicit loopback name or address."""
    if value is None:
        return False
    cleaned = value.strip().lower()
    if cleaned == "localhost":
        return True
    if allow_test_hosts and cleaned in _TEST_HOSTS:
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def require_loopback_bind_host(host: str) -> str:
    """Validate and normalize the dashboard's mandatory loopback bind host."""
    cleaned = host.strip()
    if not is_loopback_host(cleaned):
        raise ValueError(
            "dashboard is localhost-only; bind host must be localhost or an "
            "explicit loopback IP address"
        )
    return cleaned
