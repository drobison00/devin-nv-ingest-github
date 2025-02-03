import pytest

from nv_ingest.modules.telemetry.job_counter import _count_jobs
from nv_ingest.primitives.ingest_control_message import IngestControlMessage


class FakeGlobalStats:
    """
    A fake global statistics object for testing.
    """

    def __init__(self):
        self.counts = {}

    def increment_stat(self, stat_name: str) -> None:
        """
        Simulate incrementing a statistic.
        """
        self.counts[stat_name] = self.counts.get(stat_name, 0) + 1


class FakeGlobalStatsException:
    """
    A fake global statistics object that raises an exception when incrementing a statistic.
    """

    def increment_stat(self, stat_name: str) -> None:
        """
        Simulate a failure in incrementing a statistic.
        """
        raise Exception("Simulated failure in increment_stat")


def test_count_jobs_completed_success():
    """
    Validate that when 'completed_jobs' is specified and the message does not indicate failure,
    _count_jobs increments 'completed_jobs' and returns the message unchanged.
    """
    message = IngestControlMessage()
    fake_stats = FakeGlobalStats()
    result = _count_jobs(message, "completed_jobs", fake_stats)
    assert result is message
    assert fake_stats.counts.get("completed_jobs", 0) == 1
    assert "failed_jobs" not in fake_stats.counts


def test_count_jobs_completed_failure():
    """
    Validate that when 'completed_jobs' is specified but the message indicates failure (cm_failed is True),
    _count_jobs increments 'failed_jobs' instead and returns the message unchanged.
    """
    message = IngestControlMessage()
    message.set_metadata("cm_failed", True)
    fake_stats = FakeGlobalStats()
    result = _count_jobs(message, "completed_jobs", fake_stats)
    assert result is message
    assert fake_stats.counts.get("failed_jobs", 0) == 1
    assert "completed_jobs" not in fake_stats.counts


def test_count_jobs_custom_stat():
    """
    Validate that when a custom stat is specified, _count_jobs increments that statistic and returns the
    message unchanged.
    """
    message = IngestControlMessage()
    fake_stats = FakeGlobalStats()
    result = _count_jobs(message, "custom_stat", fake_stats)
    assert result is message
    assert fake_stats.counts.get("custom_stat", 0) == 1


def test_count_jobs_exception():
    """
    Validate that if an exception occurs during statistics incrementation,
    _count_jobs raises a ValueError with an appropriate error message.
    """
    message = IngestControlMessage()
    fake_stats = FakeGlobalStatsException()
    with pytest.raises(ValueError) as exc_info:
        _count_jobs(message, "any_stat", fake_stats)
    assert "Failed to run job counter" in str(exc_info.value)
