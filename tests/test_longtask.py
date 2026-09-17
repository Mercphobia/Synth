"""Tests for long-horizon task management."""

import tempfile
from pathlib import Path

import pytest

from synth.longtask import LongTask, LongTaskStore, TaskPlanner, TaskStep


def test_task_step_creation():
    """Test TaskStep dataclass creation."""
    step = TaskStep(description="Test step")
    assert step.description == "Test step"
    assert step.completed is False
    assert step.result == ""


def test_long_task_progress_percent_empty():
    """Test progress calculation with no steps."""
    task = LongTask(id=1, goal="Test goal", steps=[])
    assert task.progress_percent() == 0.0


def test_long_task_progress_percent_partial():
    """Test progress calculation with partial completion."""
    steps = [
        TaskStep("Step 1", completed=True),
        TaskStep("Step 2", completed=False),
        TaskStep("Step 3", completed=True),
    ]
    task = LongTask(id=1, goal="Test goal", steps=steps)
    assert task.progress_percent() == pytest.approx(66.67, rel=1e-2)


def test_long_task_progress_percent_complete():
    """Test progress calculation with full completion."""
    steps = [
        TaskStep("Step 1", completed=True),
        TaskStep("Step 2", completed=True),
    ]
    task = LongTask(id=1, goal="Test goal", steps=steps)
    assert task.progress_percent() == 100.0


def test_long_task_to_dict():
    """Test LongTask serialization to dictionary."""
    steps = [TaskStep("Step 1", completed=True, result="Done")]
    task = LongTask(
        id=1,
        goal="Test goal",
        steps=steps,
        created_at="2026-01-01T00:00:00",
        completed_at="2026-01-01T01:00:00",
    )
    
    data = task.to_dict()
    
    assert data["id"] == 1
    assert data["goal"] == "Test goal"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["description"] == "Step 1"
    assert data["steps"][0]["completed"] is True
    assert data["steps"][0]["result"] == "Done"


def test_long_task_from_dict():
    """Test LongTask deserialization from dictionary."""
    data = {
        "id": 1,
        "goal": "Test goal",
        "steps": [
            {
                "description": "Step 1",
                "completed": True,
                "result": "Done",
                "started_at": "",
                "completed_at": "",
            }
        ],
        "created_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T01:00:00",
    }
    
    task = LongTask.from_dict(data)
    
    assert task.id == 1
    assert task.goal == "Test goal"
    assert len(task.steps) == 1
    assert task.steps[0].description == "Step 1"
    assert task.steps[0].completed is True


def test_task_store_create_and_get():
    """Test creating and retrieving a task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.create_task(
                "Build a feature",
                ["Design", "Implement", "Test"],
            )
            
            assert task.id > 0
            assert task.goal == "Build a feature"
            assert len(task.steps) == 3
            
            retrieved = store.get_task(task.id)
            assert retrieved is not None
            assert retrieved.id == task.id
            assert retrieved.goal == task.goal
            assert len(retrieved.steps) == 3
        finally:
            store.close()


def test_task_store_get_nonexistent():
    """Test retrieving a non-existent task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.get_task(999)
            assert task is None
        finally:
            store.close()


def test_task_store_mark_step_complete():
    """Test marking a step as complete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.create_task(
                "Build feature",
                ["Design", "Implement", "Test"],
            )
            
            updated = store.mark_step_complete(task.id, 1, "Implementation done")
            
            assert updated.steps[1].completed is True
            assert updated.steps[1].result == "Implementation done"
            assert updated.steps[1].completed_at != ""
            assert updated.steps[0].completed is False
            assert updated.steps[2].completed is False
            
            # Check progress
            assert updated.progress_percent() == pytest.approx(33.33, rel=1e-2)
        finally:
            store.close()


def test_task_store_mark_all_steps_complete():
    """Test that completing all steps marks task as complete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.create_task(
                "Build feature",
                ["Step 1", "Step 2"],
            )
            
            store.mark_step_complete(task.id, 0)
            updated = store.mark_step_complete(task.id, 1)
            
            assert updated.completed_at != ""
            assert updated.progress_percent() == 100.0
        finally:
            store.close()


def test_task_store_mark_invalid_step():
    """Test marking an invalid step raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.create_task("Test", ["Step 1"])
            
            with pytest.raises(ValueError, match="Invalid step index"):
                store.mark_step_complete(task.id, 5)
        finally:
            store.close()


def test_task_store_list_tasks():
    """Test listing tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task1 = store.create_task("Task 1", ["Step 1"])
            task2 = store.create_task("Task 2", ["Step 1", "Step 2"])
            store.mark_step_complete(task1.id, 0)
            
            # List incomplete only
            incomplete = store.list_tasks(include_completed=False)
            assert len(incomplete) == 1
            assert incomplete[0].id == task2.id
            
            # List all
            all_tasks = store.list_tasks(include_completed=True)
            assert len(all_tasks) == 2
        finally:
            store.close()


def test_task_store_delete_task():
    """Test deleting a task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tasks.db"
        store = LongTaskStore(db_path)
        
        try:
            task = store.create_task("Test", ["Step 1"])
            
            assert store.delete_task(task.id) is True
            assert store.get_task(task.id) is None
            assert store.delete_task(task.id) is False
        finally:
            store.close()


def test_task_planner_numbered_list():
    """Test planner with numbered list."""
    goal = """1. Design the system
2. Implement core features
3. Write tests
4. Deploy to production"""
    
    steps = TaskPlanner.decompose(goal)
    
    assert len(steps) == 4
    assert "Design the system" in steps
    assert "Implement core features" in steps


def test_task_planner_bullet_list():
    """Test planner with bullet list."""
    goal = """- Research requirements
- Create mockups
- Build prototype"""
    
    steps = TaskPlanner.decompose(goal)
    
    assert len(steps) == 3
    assert "Research requirements" in steps
    assert "Create mockups" in steps


def test_task_planner_generic_fallback():
    """Test planner with generic goal (no structure)."""
    goal = "Build a new feature for the app"
    
    steps = TaskPlanner.decompose(goal)
    
    assert len(steps) > 0
    assert "Understand requirements" in steps
    assert "Implement solution" in steps


def test_task_planner_max_steps():
    """Test planner respects max_steps limit."""
    goal = "\n".join([f"{i}. Step {i}" for i in range(1, 20)])
    
    steps = TaskPlanner.decompose(goal, max_steps=5)
    
    assert len(steps) == 5


def test_task_planner_empty_goal():
    """Test planner with empty goal returns generic steps."""
    goal = ""
    
    steps = TaskPlanner.decompose(goal)
    
    assert len(steps) > 0
    assert "Understand requirements" in steps
