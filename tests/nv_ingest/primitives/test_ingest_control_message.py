from nv_ingest.primitives.ingest_control_message import ControlMessageTask

import pytest
from pydantic import ValidationError


def test_valid_task():
    data = {
        "name": "Example Task",
        "id": "task-123",
        "properties": {"param1": "value1", "param2": 42},
    }
    task = ControlMessageTask(**data)
    assert task.name == "Example Task"
    assert task.id == "task-123"
    assert task.properties == {"param1": "value1", "param2": 42}


def test_valid_task_without_properties():
    data = {
        "name": "Minimal Task",
        "id": "task-456",
    }
    task = ControlMessageTask(**data)
    assert task.name == "Minimal Task"
    assert task.id == "task-456"
    assert task.properties == {}


def test_missing_required_field_name():
    data = {"id": "task-no-name", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("name",)
    assert errors[0]["type"] == "missing"


def test_missing_required_field_id():
    data = {"name": "Task With No ID", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("id",)
    assert errors[0]["type"] == "missing"


def test_extra_fields_forbidden():
    data = {"name": "Task With Extras", "id": "task-extra", "properties": {}, "unexpected_field": "foo"}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("unexpected_field",)


def test_properties_accepts_various_types():
    data = {
        "name": "Complex Properties Task",
        "id": "task-complex",
        "properties": {
            "string_prop": "string value",
            "int_prop": 123,
            "list_prop": [1, 2, 3],
            "dict_prop": {"nested": True},
        },
    }
    task = ControlMessageTask(**data)
    assert task.properties["string_prop"] == "string value"
    assert task.properties["int_prop"] == 123
    assert task.properties["list_prop"] == [1, 2, 3]
    assert task.properties["dict_prop"] == {"nested": True}


def test_properties_with_invalid_type():
    data = {"name": "Invalid Properties Task", "id": "task-invalid-props", "properties": ["this", "should", "fail"]}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("properties",)
