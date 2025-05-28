import asyncio
import inspect
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import anyio
from pydantic import BaseModel

from browser_use.event_bus.views import Event

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Union[Callable[[Event, Any], Any], Callable[[Event, Any], Awaitable[Any]]]


class EventBus:
	"""
	Async event bus with write-ahead logging and guaranteed FIFO processing.

	Features:
	- Sync and async event enqueueing
	- Blocking and non-blocking dispatch
	- Write-ahead logging with UUIDs and timestamps
	- FIFO event processing
	- Parallel handler execution per event
	- Serial event processing
	"""

	def __init__(self, agent_object: Any):
		self.agent = agent_object
		self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
		self.write_ahead_log: list[Event] = []
		self.handlers: dict[str, list[EventHandler]] = defaultdict(list)
		self.all_event_handlers: list[EventHandler] = []
		self.running = False
		self.runner_task: asyncio.Task | None = None
		self.lock = asyncio.Lock()

		# Register default logger handler
		self.subscribe_to_all(self._default_log_handler)

	async def _default_log_handler(self, event: Event, agent: Any) -> str:
		"""Default handler that logs all events"""
		logger.debug(f'Event processed: {event.event_type} [{event.event_id}] - {event.model_dump_json()}')
		return 'logged'

	def subscribe(self, event_type: str, handler: EventHandler) -> None:
		"""Subscribe a handler to a specific event type"""
		if not inspect.iscoroutinefunction(handler):
			raise ValueError('Handler must be an async function')
		self.handlers[event_type].append(handler)

	def subscribe_by_model(self, event_model: type[BaseModel], handler: EventHandler) -> None:
		"""Subscribe a handler to a specific event model type"""
		self.subscribe(event_model.__name__, handler)

	def subscribe_to_all(self, handler: EventHandler) -> None:
		"""Subscribe a handler to all event types"""
		if not inspect.iscoroutinefunction(handler):
			raise ValueError('Handler must be an async function')
		self.all_event_handlers.append(handler)

	def _log_event(self, event: Event) -> Event:
		"""Log an event to the write-ahead log"""
		self.write_ahead_log.append(event)
		return event

	async def enqueue(self, event: BaseModel) -> Event:
		"""
		Enqueue an event (non-blocking).
		Returns the event object immediately with UUID but no results.
		Can be awaited to get results when processing completes.
		"""
		# Cloud events inherit from Event, so just use them directly
		if isinstance(event, Event):
			self._log_event(event)
			await self.event_queue.put(event)
			return event

		# For other BaseModels, wrap in Event
		event_data = event.model_dump()
		wrapped_event = Event(data=event_data)
		self._log_event(wrapped_event)
		await self.event_queue.put(wrapped_event)
		return wrapped_event

	def enqueue_sync(self, event: BaseModel) -> Event:
		"""
		Enqueue an event from sync context (non-blocking).
		Returns the event object immediately with UUID but no results.
		"""
		# Cloud events inherit from Event, so just use them directly
		if isinstance(event, Event):
			actual_event = event
		else:
			# For other BaseModels, wrap in Event
			event_data = event.model_dump()
			actual_event = Event(data=event_data)

		self._log_event(actual_event)

		# Get or create event loop
		try:
			loop = asyncio.get_running_loop()
			# If loop is running, schedule the coroutine
			asyncio.create_task(self.event_queue.put(actual_event))
		except RuntimeError:
			# No event loop in current thread
			pass

		return actual_event

	async def enqueue_and_wait(self, event: BaseModel) -> Event:
		"""
		Enqueue an event and wait for all handlers to complete (blocking).
		Returns the event object with results included.
		"""
		event_obj = await self.enqueue(event)
		await event_obj.wait_for_completion()
		return event_obj

	def enqueue_and_wait_sync(self, event: BaseModel) -> Event:
		"""
		Enqueue an event and wait for all handlers to complete from sync context (blocking).
		Returns the event object with results included.
		"""
		# Check if there's a running event loop
		try:
			loop = asyncio.get_running_loop()
			# If we get here, we're in an async context
			raise RuntimeError('Cannot call enqueue_and_wait_sync from within running event loop. Use enqueue_and_wait instead.')
		except RuntimeError as e:
			# Check the specific error message
			error_msg = str(e).lower()
			if 'no running loop' in error_msg or 'no current event loop' in error_msg or 'no running event loop' in error_msg:
				# No event loop in current thread, create a new one
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					return loop.run_until_complete(self.enqueue_and_wait(event))
				finally:
					# Clean up the loop
					loop.close()
					asyncio.set_event_loop(None)
			else:
				# Some other error, re-raise
				raise

	async def enqueue_batch_and_wait(self, events: list[BaseModel]) -> list[Event]:
		"""
		Enqueue a list of events and wait for all of them to complete (blocking).
		Events are processed in FIFO order but this method waits for all to finish.
		Returns a list of completed event objects with results included.
		"""
		if not events:
			return []

		# Enqueue all events (non-blocking)
		event_objects = []
		for event in events:
			event_obj = await self.enqueue(event)
			event_objects.append(event_obj)

		# Wait for all events to complete
		await asyncio.gather(*[event_obj.wait_for_completion() for event_obj in event_objects])

		return event_objects

	def enqueue_batch_and_wait_sync(self, events: list[BaseModel]) -> list[Event]:
		"""
		Enqueue a list of events and wait for all of them to complete from sync context (blocking).
		Events are processed in FIFO order but this method waits for all to finish.
		Returns a list of completed event objects with results included.
		"""
		# Check if there's a running event loop
		try:
			loop = asyncio.get_running_loop()
			# If we get here, we're in an async context
			raise RuntimeError(
				'Cannot call enqueue_batch_and_wait_sync from within running event loop. Use enqueue_batch_and_wait instead.'
			)
		except RuntimeError as e:
			# Check the specific error message
			error_msg = str(e).lower()
			if 'no running loop' in error_msg or 'no current event loop' in error_msg or 'no running event loop' in error_msg:
				# No event loop in current thread, create a new one
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					return loop.run_until_complete(self.enqueue_batch_and_wait(events))
				finally:
					# Clean up the loop
					loop.close()
					asyncio.set_event_loop(None)
			else:
				# Some other error, re-raise
				raise

	async def _execute_handlers(self, event: Event) -> None:
		"""Execute all handlers for an event in parallel"""
		# Get all applicable handlers
		applicable_handlers = []

		# Add type-specific handlers
		applicable_handlers.extend(self.handlers.get(event.event_type, []))

		# Add all-event handlers
		applicable_handlers.extend(self.all_event_handlers)

		if not applicable_handlers:
			return

		# Execute all handlers in parallel
		tasks = []
		for handler in applicable_handlers:
			task = asyncio.create_task(self._safe_execute_handler(handler, event))
			tasks.append((handler.__name__, task))

		# Wait for all handlers to complete
		for handler_name, task in tasks:
			try:
				result = await task
				event.results[handler_name] = result
			except Exception as e:
				event.errors[handler_name] = e
				logger.error(f'Handler {handler_name} failed for event {event.event_id}: {e}')

	async def _safe_execute_handler(self, handler: EventHandler, event: Event) -> Any:
		"""Safely execute a single handler"""
		try:
			return await handler(event, self.agent)
		except Exception as e:
			logger.exception(f'Error in handler {handler.__name__} for event {event.event_id}')
			raise

	async def _step(self) -> None:
		"""Process a single event from the queue"""
		try:
			# Get next event (this will block if queue is empty)
			event = await self.event_queue.get()

			# Record start time
			event.started_at = datetime.utcnow()

			# Execute all handlers for this event
			await self._execute_handlers(event)

			# Mark event as completed
			event.mark_completed()

			# Mark task as done
			self.event_queue.task_done()

		except Exception as e:
			logger.exception(f'Error processing event: {e}')

	async def _run_loop(self) -> None:
		"""Main event processing loop"""
		while self.running:
			try:
				async with self.lock:
					await self._step()
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.exception(f'Error in event loop: {e}')
				# Continue running even if there's an error

	async def start(self) -> None:
		"""Start the event bus processing loop"""
		if self.running:
			return

		self.running = True
		self.runner_task = asyncio.create_task(self._run_loop())
		logger.info('EventBus started')

	async def stop(self) -> None:
		"""Stop the event bus processing loop"""
		if not self.running:
			return

		self.running = False

		if self.runner_task:
			self.runner_task.cancel()
			try:
				await self.runner_task
			except asyncio.CancelledError:
				pass

		logger.info('EventBus stopped')

	async def wait_for_empty_queue(self) -> None:
		"""Wait for all queued events to be processed"""
		await self.event_queue.join()

	def get_event_log(self) -> list[Event]:
		"""Get the write-ahead log of all events"""
		return self.write_ahead_log.copy()

	async def serialize_events_to_file(self, file_path: Path | str) -> None:
		"""Serialize all events to a JSON file

		Args:
			file_path: Path to save the events to
		"""
		file_path = Path(file_path)
		events_data = []

		for event in self.write_ahead_log:
			event_dict = event.model_dump()
			# Convert datetime objects to ISO format strings
			for key in ['timestamp', 'started_at', 'completed_at']:
				if key in event_dict and event_dict[key]:
					event_dict[key] = event_dict[key].isoformat()

			# Convert exception objects to strings in errors dict
			if 'errors' in event_dict:
				event_dict['errors'] = {k: str(v) for k, v in event_dict['errors'].items()}

			events_data.append(event_dict)

		async with asyncio.Lock():
			async with await anyio.open_file(file_path, 'w') as f:
				await f.write(json.dumps(events_data, indent=2, default=str))

	def serialize_events_to_file_sync(self, file_path: Path | str) -> None:
		"""Serialize all events to a JSON file (sync version)

		Args:
			file_path: Path to save the events to
		"""
		file_path = Path(file_path)
		events_data = []

		for event in self.write_ahead_log:
			event_dict = event.model_dump()
			# Convert datetime objects to ISO format strings
			for key in ['timestamp', 'started_at', 'completed_at']:
				if key in event_dict and event_dict[key]:
					event_dict[key] = event_dict[key].isoformat()

			# Convert exception objects to strings in errors dict
			if 'errors' in event_dict:
				event_dict['errors'] = {k: str(v) for k, v in event_dict['errors'].items()}

			events_data.append(event_dict)

		with open(file_path, 'w') as f:
			json.dump(events_data, f, indent=2, default=str)

	# Convenience method that matches the old API
	def emit(self, event: BaseModel) -> Event:
		"""Queue an event for processing. Can be called from sync or async context."""
		return self.enqueue_sync(event)
