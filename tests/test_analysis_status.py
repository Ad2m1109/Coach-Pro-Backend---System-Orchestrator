import os
import sys

# Ensure backend/src is importable for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import analysis_status


def test_normalize_status_maps_queued_and_pending():
    assert analysis_status.normalize_status("queued") == analysis_status.ANALYSIS_STATUS_QUEUED
    assert analysis_status.normalize_status("PENDING") == analysis_status.ANALYSIS_STATUS_QUEUED


def test_normalize_status_maps_processing_variants():
    assert analysis_status.normalize_status("processing") == analysis_status.ANALYSIS_STATUS_PROCESSING
    assert analysis_status.normalize_status("streaming") == analysis_status.ANALYSIS_STATUS_PROCESSING
    assert analysis_status.normalize_status("receiving") == analysis_status.ANALYSIS_STATUS_PROCESSING


def test_normalize_status_maps_completed_and_done():
    assert analysis_status.normalize_status("completed") == analysis_status.ANALYSIS_STATUS_COMPLETED
    assert analysis_status.normalize_status("done") == analysis_status.ANALYSIS_STATUS_COMPLETED


def test_normalize_status_maps_errors_to_failed():
    assert analysis_status.normalize_status("failed") == analysis_status.ANALYSIS_STATUS_FAILED
    assert analysis_status.normalize_status("error") == analysis_status.ANALYSIS_STATUS_FAILED


def test_terminal_statuses_set():
    assert analysis_status.ANALYSIS_STATUS_COMPLETED in analysis_status.TERMINAL_STATUSES
    assert analysis_status.ANALYSIS_STATUS_FAILED in analysis_status.TERMINAL_STATUSES
