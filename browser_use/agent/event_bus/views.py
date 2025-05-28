from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class Event(BaseModel):
	"""Base event class for all event types"""

	model_config = ConfigDict(extra='forbid', validate_assignment=True)

	event_id: str = Field(default_factory=uuid7str)
	event_type: str
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	data: dict[str, Any] = Field(default_factory=dict)


# Agent lifecycle events
class AgentStartedEvent(Event):
	"""Emitted when agent starts"""

	event_type: str = Field(default='agent_started', frozen=True)
	agent_id: str
	task: str


class AgentStoppedEvent(Event):
	"""Emitted when agent stops"""

	event_type: str = Field(default='agent_stopped', frozen=True)
	agent_id: str
	reason: str | None = None
	success: bool = True


# Step execution events
class StepStartedEvent(Event):
	"""Emitted when a step starts execution"""

	event_type: str = Field(default='step_started', frozen=True)
	step_id: str = Field(default_factory=uuid7str)
	step_number: int
	action_type: str
	params: dict[str, Any] = Field(default_factory=dict)


class StepCompletedEvent(Event):
	"""Emitted when a step completes successfully"""

	event_type: str = Field(default='step_completed', frozen=True)
	step_id: str
	step_number: int
	result: Any | None = None
	duration_ms: float


class StepFailedEvent(Event):
	"""Emitted when a step fails"""

	event_type: str = Field(default='step_failed', frozen=True)
	step_id: str
	step_number: int
	error: str
	error_type: str | None = None
	retry_count: int = 0


# State change events
class StateChangedEvent(Event):
	"""Emitted when agent state changes"""

	event_type: str = Field(default='state_changed', frozen=True)
	old_state: str | None = None
	new_state: str
	reason: str | None = None


class PageNavigatedEvent(Event):
	"""Emitted when browser navigates to new page"""

	event_type: str = Field(default='page_navigated', frozen=True)
	url: str
	title: str | None = None
	load_time_ms: float | None = None


class ElementInteractedEvent(Event):
	"""Emitted when agent interacts with DOM element"""

	event_type: str = Field(default='element_interacted', frozen=True)
	element_id: int
	interaction_type: str  # click, type, select, etc.
	selector: str | None = None
	value: Any | None = None


# Error events
class ErrorOccurredEvent(Event):
	"""Emitted when an error occurs"""

	event_type: str = Field(default='error_occurred', frozen=True)
	error: str
	error_type: str
	context: dict[str, Any] = Field(default_factory=dict)
	recoverable: bool = True


class TimeoutEvent(Event):
	"""Emitted when an operation times out"""

	event_type: str = Field(default='timeout', frozen=True)
	operation: str
	timeout_seconds: float
	context: dict[str, Any] = Field(default_factory=dict)


# Model/LLM events
class ModelCallStartedEvent(Event):
	"""Emitted when LLM call starts"""

	event_type: str = Field(default='model_call_started', frozen=True)
	model_name: str
	prompt_tokens: int | None = None
	messages_count: int | None = None


class ModelCallCompletedEvent(Event):
	"""Emitted when LLM call completes"""

	event_type: str = Field(default='model_call_completed', frozen=True)
	model_name: str
	response_tokens: int | None = None
	total_tokens: int | None = None
	duration_ms: float
	cost: float | None = None
