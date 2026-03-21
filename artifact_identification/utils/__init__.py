"""Utility modules for EDF file inspection."""

from .check_channels import check_edf_channels
from .check_edf import inspect_edf_properties

__all__ = ['check_edf_channels', 'inspect_edf_properties']
