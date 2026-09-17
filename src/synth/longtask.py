"""Long-horizon task management for multi-step workflows (spec 6.11).

Supports breaking down complex goals into trackable steps with progress
monitoring and state persistence across sessions.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR


@dataclass
class TaskStep:
    """A single step in a long-horizon task."""
    
    description: str
    completed: bool = False
    result: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class LongTask:
    """A long-horizon task with multiple steps."""
    
    id: int
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    created_at: str = ""
    completed_at: str = ""
    
    def progress_percent(self) -> float:
        """Calculate completion percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.completed)
        return (completed / len(self.steps)) * 100
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [
                {
                    "description": s.description,
                    "completed": s.completed,
                    "result": s.result,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                }
                for s in self.steps
            ],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTask":
        """Create from dictionary."""
        steps = [
            TaskStep(
                description=s["description"],
                completed=s.get("completed", False),
                result=s.get("result", ""),
                started_at=s.get("started_at", ""),
                completed_at=s.get("completed_at", ""),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            id=data["id"],
            goal=data["goal"],
            steps=steps,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
        )


class LongTaskStore:
    """Manages long-horizon task persistence."""
    
    def __init__(self, db_path: Path | None = None):
        """Initialize the task store.
        
        Args:
            db_path: Path to SQLite database. Defaults to CONFIG_DIR/longtasks.db
        """
        if db_path is None:
            db_path = Path(CONFIG_DIR) / "longtasks.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS long_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        self._conn.commit()
    
    def create_task(self, goal: str, steps: list[str]) -> LongTask:
        """Create a new long-horizon task.
        
        Args:
            goal: The overall goal description
            steps: List of step descriptions
            
        Returns:
            The created LongTask
        """
        now = datetime.now().isoformat()
        task_steps = [TaskStep(description=s) for s in steps]
        
        task = LongTask(
            id=0,  # Will be set after insert
            goal=goal,
            steps=task_steps,
            created_at=now,
        )
        
        cursor = self._conn.execute(
            "INSERT INTO long_tasks (goal, data, created_at) VALUES (?, ?, ?)",
            (goal, json.dumps(task.to_dict()), now),
        )
        self._conn.commit()
        
        task.id = cursor.lastrowid
        return task
    
    def get_task(self, task_id: int) -> LongTask | None:
        """Get a task by ID.
        
        Args:
            task_id: The task ID
            
        Returns:
            The LongTask or None if not found
        """
        cursor = self._conn.execute(
            "SELECT id, goal, data, created_at, completed_at FROM long_tasks WHERE id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        data = json.loads(row["data"])
        task = LongTask.from_dict(data)
        task.id = row["id"]
        task.created_at = row["created_at"]
        task.completed_at = row["completed_at"]
        
        return task
    
    def update_task(self, task: LongTask) -> None:
        """Update a task in the database.
        
        Args:
            task: The task to update
        """
        self._conn.execute(
            "UPDATE long_tasks SET data = ?, completed_at = ? WHERE id = ?",
            (json.dumps(task.to_dict()), task.completed_at, task.id),
        )
        self._conn.commit()
    
    def mark_step_complete(
        self, task_id: int, step_index: int, result: str = ""
    ) -> LongTask:
        """Mark a step as completed.
        
        Args:
            task_id: The task ID
            step_index: Index of the step to complete
            result: Optional result description
            
        Returns:
            The updated LongTask
            
        Raises:
            ValueError: If task or step not found
        """
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        
        if step_index < 0 or step_index >= len(task.steps):
            raise ValueError(f"Invalid step index {step_index}")
        
        now = datetime.now().isoformat()
        task.steps[step_index].completed = True
        task.steps[step_index].result = result
        task.steps[step_index].completed_at = now
        
        # Check if all steps complete
        if all(s.completed for s in task.steps):
            task.completed_at = now
        
        self.update_task(task)
        return task
    
    def list_tasks(self, include_completed: bool = False) -> list[LongTask]:
        """List all tasks.
        
        Args:
            include_completed: Whether to include completed tasks
            
        Returns:
            List of LongTask objects
        """
        if include_completed:
            cursor = self._conn.execute(
                "SELECT id, goal, data, created_at, completed_at FROM long_tasks ORDER BY id DESC"
            )
        else:
            cursor = self._conn.execute(
                "SELECT id, goal, data, created_at, completed_at FROM long_tasks WHERE completed_at IS NULL ORDER BY id DESC"
            )
        
        tasks = []
        for row in cursor:
            data = json.loads(row["data"])
            task = LongTask.from_dict(data)
            task.id = row["id"]
            task.created_at = row["created_at"]
            task.completed_at = row["completed_at"]
            tasks.append(task)
        
        return tasks
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self._conn.execute(
            "DELETE FROM long_tasks WHERE id = ?", (task_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()


class TaskPlanner:
    """Breaks down goals into actionable steps using heuristics."""
    
    @staticmethod
    def decompose(goal: str, max_steps: int = 10) -> list[str]:
        """Break a goal into steps using simple heuristics.
        
        Detects numbered lists (1. 2. 3.) and bullet points (- or *).
        Falls back to generic steps if no structure detected.
        
        Args:
            goal: The goal description
            max_steps: Maximum number of steps to return
            
        Returns:
            List of step descriptions
        """
        lines = [l.strip() for l in goal.split("\n") if l.strip()]
        
        # Check for numbered list
        if any(l[0].isdigit() and "." in l for l in lines[:5]):
            steps = []
            for line in lines[:max_steps]:
                # Remove numbering
                if "." in line:
                    step = line.split(".", 1)[1].strip()
                    steps.append(step)
            if steps:
                return steps
        
        # Check for bullet points
        if any(l.startswith("-") or l.startswith("*") for l in lines[:5]):
            steps = []
            for line in lines[:max_steps]:
                if line.startswith("-") or line.startswith("*"):
                    step = line[1:].strip()
                    steps.append(step)
            if steps:
                return steps
        
        # Fallback: generic steps
        return [
            "Understand requirements",
            "Plan approach",
            "Implement solution",
            "Test and validate",
            "Document results",
        ][:max_steps]
