"""Automated operations — system monitoring, alerting, auto-remediation.

Self-contained module for edge device health monitoring.
Uses ``psutil`` + Jetson-specific sensors (via ``jtop`` or ``tegrastats``)
when available, falls back to basic metrics on non-Jetson platforms.
"""
