import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

from browser_use.agent.event_bus.views import Event

EventHandler = Callable[[Event], None] | Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
	"""Event bus for managing and dispatching events"""

	def __init__(self, store_events: bool = True):
		self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
		self._all_handlers: list[EventHandler] = []
		self._events: list[Event] = []
		self._store_events = store_events
		self._lock = asyncio.Lock()
		self._pending_tasks: set[asyncio.Task] = set()

	def register_handler(self, event_type: str | None, handler: EventHandler) -> None:
		"""Register a handler for specific event type or all events

		Args:
			event_type: Event type to listen for, or None for all events
			handler: Sync or async function to handle the event
		"""
		if event_type is None:
			self._all_handlers.append(handler)
		else:
			self._handlers[event_type].append(handler)

	def unregister_handler(self, event_type: str | None, handler: EventHandler) -> None:
		"""Unregister a handler

		Args:
			event_type: Event type the handler was registered for
			handler: Handler to remove
		"""
		if event_type is None:
			if handler in self._all_handlers:
				self._all_handlers.remove(handler)
		else:
			if handler in self._handlers[event_type]:
				self._handlers[event_type].remove(handler)

	def emit(self, event: Event) -> None:
		"""Queue an event for processing. Can be called from sync or async context.

		This method queues the event and, if in an async context, schedules
		the handlers to run asynchronously. If in a sync context, the event
		is only queued and handlers will run when the event loop is available.

		Args:
			event: Event to emit
		"""
		# Always store event if enabled
		if self._store_events:
			self._events.append(event)

		# Try to get the current event loop
		try:
			loop = asyncio.get_running_loop()
			# We're in an async context, schedule the async emit
			task = asyncio.create_task(self._process_event(event))
			# Store the task to prevent it from being garbage collected
			self._pending_tasks.add(task)
			task.add_done_callback(self._pending_tasks.discard)
		except RuntimeError:
			# No event loop running, event is queued for later processing
			pass

	async def _process_event(self, event: Event) -> None:
		"""Process a single event by calling all its handlers"""
		# Collect all relevant handlers
		handlers = list(self._all_handlers)
		if event.event_type in self._handlers:
			handlers.extend(self._handlers[event.event_type])

		# Execute handlers
		tasks = []
		for handler in handlers:
			if asyncio.iscoroutinefunction(handler):
				tasks.append(self._execute_async_handler(handler, event))
			else:
				tasks.append(self._execute_sync_handler(handler, event))

		# Wait for all handlers to complete
		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)

	async def _execute_async_handler(self, handler: EventHandler, event: Event) -> None:
		"""Execute async handler with error handling"""
		try:
			await handler(event)
		except Exception as e:
			self._log_handler_error(handler, event, e)

	async def _execute_sync_handler(self, handler: EventHandler, event: Event) -> None:
		"""Execute sync handler in thread pool with error handling"""
		try:
			loop = asyncio.get_event_loop()
			await loop.run_in_executor(None, handler, event)
		except Exception as e:
			self._log_handler_error(handler, event, e)

	def _log_handler_error(self, handler: EventHandler, event: Event, error: Exception) -> None:
		"""Log handler execution error"""
		handler_name = getattr(handler, '__name__', str(handler))
		print(f"Error in event handler '{handler_name}' for event {event.event_type}: {error}")

	def get_events(self) -> list[Event]:
		"""Get all stored events"""
		return list(self._events)

	def clear_events(self) -> None:
		"""Clear stored events"""
		self._events.clear()

	async def serialize_events_to_file(self, file_path: Path | str) -> None:
		"""Serialize all events to a JSON file

		Args:
			file_path: Path to write events to
		"""
		file_path = Path(file_path)

		# Ensure parent directory exists
		file_path.parent.mkdir(parents=True, exist_ok=True)

		# Convert events to dicts
		events_data = []
		async with self._lock:
			for event in self._events:
				event_dict = event.model_dump(mode='json')
				# Ensure datetime is serialized properly
				if 'timestamp' in event_dict and isinstance(event_dict['timestamp'], datetime):
					event_dict['timestamp'] = event_dict['timestamp'].isoformat()
				events_data.append(event_dict)

		# Write to file
		with open(file_path, 'w') as f:
			json.dump(events_data, f, indent=2, default=str)

		self._log_serialization_complete(file_path, len(events_data))

	def _log_serialization_complete(self, file_path: Path, event_count: int) -> None:
		"""Log successful serialization"""
		print(f'Serialized {event_count} events to {file_path}')

	async def load_events_from_file(self, file_path: Path | str) -> list[Event]:
		"""Load events from a JSON file

		Args:
			file_path: Path to read events from

		Returns:
			List of loaded events
		"""
		file_path = Path(file_path)

		if not file_path.exists():
			raise FileNotFoundError(f'Event file not found: {file_path}')

		with open(file_path) as f:
			events_data = json.load(f)

		# Convert back to Event objects
		loaded_events = []
		for event_dict in events_data:
			# Parse timestamp if it's a string
			if 'timestamp' in event_dict and isinstance(event_dict['timestamp'], str):
				event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])

			# Create generic Event - in real usage, you'd map event_type to specific classes
			event = Event(**event_dict)
			loaded_events.append(event)

		return loaded_events

	def decorator(self, event_type: str | None = None):
		"""Decorator for registering event handlers

		Usage:
			@event_bus.decorator('step_started')
			async def on_step_started(event: StepStartedEvent):
				print(f"Step {event.step_number} started")
		"""

		def wrapper(handler: EventHandler) -> EventHandler:
			self.register_handler(event_type, handler)
			return handler

		return wrapper
