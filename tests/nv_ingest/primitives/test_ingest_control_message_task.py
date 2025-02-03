import pytest

from nv_ingest.primitives.ingest_control_message import IngestControlMessage, ControlMessageTask


def test_empty_control_message():
    cm = IngestControlMessage()
    assert list(cm.get_tasks()) == []
    assert not cm.has_task("nonexistent")


def test_add_single_task():
    cm = IngestControlMessage()
    task = ControlMessageTask(name="Test Task", id="task1", properties={"key": "value"})
    cm.add_task(task)
    assert cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert len(tasks) == 1
    assert tasks[0] == task


def test_add_duplicate_task():
    cm = IngestControlMessage()
    task = ControlMessageTask(name="Test Task", id="task1", properties={"key": "value"})
    cm.add_task(task)
    duplicate_task = ControlMessageTask(name="Another Task", id="task1", properties={"key": "other"})
    with pytest.raises(ValueError) as exc_info:
        cm.add_task(duplicate_task)
    assert "already exists" in str(exc_info.value)


def test_multiple_tasks():
    cm = IngestControlMessage()
    task_data = [
        {"name": "Task A", "id": "a", "properties": {}},
        {"name": "Task B", "id": "b", "properties": {"x": 10}},
        {"name": "Task C", "id": "c", "properties": {"y": 20}},
    ]
    tasks = [ControlMessageTask(**data) for data in task_data]
    for task in tasks:
        cm.add_task(task)
    for data in task_data:
        assert cm.has_task(data["id"])
    retrieved_tasks = list(cm.get_tasks())
    assert len(retrieved_tasks) == len(task_data)
    retrieved_ids = {t.id for t in retrieved_tasks}
    expected_ids = {data["id"] for data in task_data}
    assert retrieved_ids == expected_ids
