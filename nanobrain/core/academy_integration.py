"""Academy Integration for Nanobrain Workflows.

This module provides the plumbing that connects Nanobrain workflows to
the Academy agent framework (``academy-py``). It exposes:

- ``AcademyNotImplementedError`` — raised when Academy-backed paths are
  invoked without real Academy integration wired up.
- ``AcademyManagerWrapper`` — process-level singleton that owns a real
  ``academy.manager.Manager`` context, hosts ``academy.handle.Handle``
  objects by human-readable name, and dispatches action calls through
  them.
- ``AcademyAgentHandle`` — thin wrapper around a real Academy Handle that
  adds Nanobrain-side logging and honors the ``ACADEMY_DEMO_MODE=1`` mock
  fallback.
- ``AcademyIntegration`` — utility class (``requires_academy_integration``,
  ``setup_academy_manager``, ``get_academy_manager``).
- ``shutdown_academy_manager`` — free function for clean teardown (tests).

## T14 mocks-policy history + G5 fix (2026-04-24)

Before T14 (2026-04-23), ``AcademyAgentHandle.__call__`` silently returned
synthetic ``_generate_mock_response`` data regardless of whether any
Academy agents were deployed. The literal shipped comment was:

    # Real Academy agent call would go here
    # For now, always use mock response

T14 hardened this to raise ``AcademyNotImplementedError`` by default,
with ``ACADEMY_DEMO_MODE=1`` as an explicit opt-in to preserve the mock
for existing demos.

G5 (2026-04-24) closes the real integration: in the non-demo path,
``AcademyAgentHandle.__call__`` now dispatches through a registered
``academy.handle.Handle`` via the canonical ``async with Manager(...)``
lifecycle. The prior implementation of ``_ensure_manager`` never entered
the manager's context, which meant ``Handle.shutdown`` / ``Handle``
action calls raised ``ExchangeClientNotFoundError`` — the real path
was not actually executable. This module now enters and exits the
Manager context explicitly and provides ``shutdown_academy_manager()``
for clean teardown.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Union

from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)

# Process-level Academy manager singleton.
_GLOBAL_ACADEMY_MANAGER: Optional["AcademyManagerWrapper"] = None


class AcademyNotImplementedError(NotImplementedError):
    """Raised when Academy-backed code paths cannot be served.

    Two causes:
    - ``ACADEMY_DEMO_MODE`` is not set AND no real Academy handle has
      been registered for the requested agent name. The caller needs to
      either opt in to the mock (``ACADEMY_DEMO_MODE=1``) or register
      the real agent via
      ``AcademyManagerWrapper.register_agent_class(name, cls)``.
    - An action name is requested on an agent that does not define it.
    """


class AcademyAgentHandle:
    """Nanobrain-side wrapper around a real ``academy.handle.Handle``.

    Provides:
    - Structured logging on every action call.
    - ``ACADEMY_DEMO_MODE=1`` mock fallback (preserved for demos).
    - A clear error message when the real path is taken but no Academy
      Handle has been registered for this agent name.

    Direct attribute access (``handle.some_action``) returns a coroutine
    factory so callers can write ``await handle.some_action(data)`` the
    same way they would with a plain ``academy.handle.Handle``. The
    attribute-access path is what ``AcademyLink._call_remote_action`` /
    ``AcademyAgentStep._invoke_with_retry`` exercise.
    """

    def __init__(
        self,
        agent_name: str,
        manager_wrapper: "AcademyManagerWrapper",
        real_handle: Any = None,
    ) -> None:
        self.agent_name = agent_name
        self.manager_wrapper = manager_wrapper
        # Real academy.handle.Handle, populated by
        # AcademyManagerWrapper.register_agent_class. None means the
        # caller has not yet registered the real agent.
        self.real_handle = real_handle
        self.logger = get_logger(f"AcademyAgentHandle.{agent_name}")

    async def __call__(self, action_name: str, *args, **kwargs):
        """Dispatch ``agent_name.action_name(*args, **kwargs)``.

        Order of precedence:
        1. ``ACADEMY_DEMO_MODE=1`` → synthesized mock response. Log line
           is ``warning`` (not ``info``) so operators can't miss it.
        2. Registered real handle → dispatch via the real handle.
        3. Neither → raise ``AcademyNotImplementedError``.

        Critical: the unregistered-agent FAIL-FAST is checked BEFORE
        any Manager interaction. Pre-2026-05-09 the code entered the
        Manager via ``_ensure_manager()`` first, which made the
        placeholder-dispatch path leak a partially-initialized Manager
        on raise — the next test in the same process then deadlocked
        on ``Manager.__aenter__``. The reordering eliminates that
        cross-test contamination shape.
        """
        # FAST PATH 1: demo mode is opt-in via env var. No Manager
        # interaction needed; the mock response uses purely local data.
        if os.environ.get("ACADEMY_DEMO_MODE") == "1":
            self.logger.info(
                f"🚀 Calling Academy agent {self.agent_name}.{action_name}() "
                "(demo mode)"
            )
            self.logger.warning(
                f"⚠️  ACADEMY_DEMO_MODE=1: returning mock response for "
                f"Academy agent {self.agent_name}.{action_name}(). "
                "Set ACADEMY_DEMO_MODE=0 (or unset it) and register the "
                "real agent via AcademyManagerWrapper.register_agent_class"
                " to exercise the real path."
            )
            return self._generate_mock_response(action_name, args, kwargs)

        # FAST PATH 2: placeholder handle (no real agent registered)
        # MUST fail-fast WITHOUT touching the Manager — see the
        # cross-test-contamination warning in the docstring above.
        if self.real_handle is None:
            raise AcademyNotImplementedError(
                f"Academy agent {self.agent_name!r} has no real Handle "
                "registered. Call "
                "AcademyManagerWrapper.register_agent_class(name, cls) "
                "to launch and register the agent, or set "
                "ACADEMY_DEMO_MODE=1 to opt in to the mock."
            )

        # Real path: Manager must be ensured for action dispatch via the
        # underlying academy.handle.Handle.
        await self.manager_wrapper._ensure_manager()

        self.logger.info(
            f"🚀 Calling Academy agent {self.agent_name}.{action_name}()"
        )

        # Academy's Handle.__getattr__ returns a remote-method-call
        # wrapper for any name — so getattr always succeeds. An action
        # that does not exist raises AttributeError from the remote
        # runtime ("Agent<...> does not have an action named '<name>'")
        # when the awaitable is awaited. We intentionally let that
        # propagate unwrapped: Academy's error is already explicit, and
        # a real internal error in the agent's action code also raises
        # AttributeError — blanket-converting it to
        # AcademyNotImplementedError would mislabel legitimate bugs as
        # "integration incomplete".
        real_action = getattr(self.real_handle, action_name)
        return await real_action(*args, **kwargs)

    def __getattr__(self, name: str):
        """Expose ``handle.action_name(*args, **kwargs)`` as a call to
        ``__call__(name, *args, **kwargs)``.

        This preserves the existing call style used by
        ``AcademyLink._call_remote_action`` and
        ``AcademyAgentStep._invoke_with_retry``.
        """
        # Only reached for names NOT already set on the instance (avoids
        # recursion with ``agent_name`` / ``real_handle`` etc.).
        if name.startswith("_"):
            raise AttributeError(name)

        async def _invoke(*args, **kwargs):
            return await self.__call__(name, *args, **kwargs)

        return _invoke

    # ------------------------------------------------------------------
    # Demo-mode mock responses (preserved from pre-T14 behavior so the
    # aurora demo keeps working).
    # ------------------------------------------------------------------

    def _generate_mock_response(self, action_name: str, args, kwargs):
        """Synthesize a mock response for demo-mode dispatch."""
        input_data = None
        if args:
            input_data = args[0]
        elif "input_data" in kwargs:
            input_data = kwargs["input_data"]
        elif "data" in kwargs:
            input_data = kwargs["data"]

        if action_name == "process":
            if self.agent_name == "aurora_computation_agent":
                sequences = []
                if isinstance(input_data, dict) and "prepared_sequences" in input_data:
                    sequences = input_data["prepared_sequences"]
                    self.logger.info(
                        f"✅ Extracted {len(sequences)} prepared sequences from input"
                    )
                return {"prepared_sequences": sequences}

            if self.agent_name == "aurora_results_agent":
                computed_sequences = []
                computation_metadata: Dict[str, Any] = {}
                node_information: Dict[str, Any] = {}
                if isinstance(input_data, dict):
                    if "computed_sequences" in input_data:
                        computed_sequences = input_data["computed_sequences"]
                    if "computation_metadata" in input_data:
                        computation_metadata = input_data["computation_metadata"]
                    if "node_information" in input_data:
                        node_information = input_data["node_information"]
                return {
                    "computed_sequences": computed_sequences,
                    "computation_metadata": computation_metadata,
                    "node_information": node_information,
                }

            return {
                "status": "success",
                "message": f"Mock response for {self.agent_name}.{action_name}",
                "data": input_data,
            }

        return {
            "status": "success",
            "message": f"Mock response for {action_name}",
            "data": input_data,
        }


class AcademyManagerWrapper:
    """Owns the real ``academy.manager.Manager`` context + a handle registry.

    Lifecycle:
    - First call to ``_ensure_manager`` enters the Manager's async
      context (``async with`` via ``__aenter__``) and holds it for the
      process lifetime.
    - ``register_agent_class(name, cls)`` launches a real Academy agent
      of the given class and stores the returned ``academy.handle.Handle``
      in the registry under ``name``.
    - ``get_handle(name)`` returns an ``AcademyAgentHandle`` wrapper for
      the registered agent (creating the wrapper lazily on first call).
    - ``shutdown()`` shuts down all registered agents and exits the
      Manager context. Idempotent.

    Use the process-level singleton via
    ``AcademyIntegration.setup_academy_manager()`` /
    ``AcademyIntegration.get_academy_manager()``; callers outside of
    tests should not instantiate this class directly.
    """

    def __init__(self) -> None:
        # Deferred import: ``academy`` is an optional install. Keep the
        # import inside the method so modules that don't touch Academy
        # do not pay the cost / hard-fail on missing dep.
        from academy.exchange import LocalExchangeFactory  # noqa: F401

        self._manager = None
        self._manager_cm = None  # the context-manager object
        self._real_handles: Dict[str, Any] = {}  # name -> academy.handle.Handle
        self._wrappers: Dict[str, AcademyAgentHandle] = {}
        self._launched_handles: list = []  # for orderly shutdown
        self._executor: Optional[ThreadPoolExecutor] = None
        self.logger = get_logger("AcademyManagerWrapper")

    async def _ensure_manager(self) -> None:
        """Enter the Academy Manager's async context on first use."""
        if self._manager is not None:
            return

        # Deferred imports (academy is optional).
        from academy.exchange import LocalExchangeFactory
        from academy.manager import Manager

        self.logger.info("🔄 Entering Academy Manager context (singleton)")
        factory = LocalExchangeFactory()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._manager_cm = await Manager.from_exchange_factory(
            factory=factory,
            executors=self._executor,
        )
        # Enter the async context explicitly so we can hold it for the
        # process lifetime. Paired with ``shutdown()`` below.
        self._manager = await self._manager_cm.__aenter__()
        self.logger.info("✅ Academy Manager entered successfully")

    async def register_agent_class(
        self, agent_name: str, agent_class: Any
    ) -> "AcademyAgentHandle":
        """Launch an Academy agent of ``agent_class`` and register its Handle
        under ``agent_name``.

        Returns the nanobrain-side ``AcademyAgentHandle`` wrapper for the
        launched agent, ready to dispatch actions via
        ``await wrapper.action_name(data)``.

        Idempotent per agent_name — a second call for the same name
        returns the existing wrapper without launching again.
        """
        if agent_name in self._wrappers:
            return self._wrappers[agent_name]

        await self._ensure_manager()

        self.logger.info(f"🚀 Launching Academy agent {agent_name!r} ({agent_class.__name__})")
        real_handle = await self._manager.launch(agent_class)
        self._real_handles[agent_name] = real_handle
        self._launched_handles.append(real_handle)

        wrapper = AcademyAgentHandle(
            agent_name=agent_name,
            manager_wrapper=self,
            real_handle=real_handle,
        )
        self._wrappers[agent_name] = wrapper
        self.logger.info(f"📝 Registered Academy agent {agent_name!r}")
        return wrapper

    def register_agent_handle(
        self, agent_name: str, real_handle: Any
    ) -> "AcademyAgentHandle":
        """Register an already-launched ``academy.handle.Handle`` under
        ``agent_name``.

        Use this when the agent was launched elsewhere (another process,
        another exchange) and you only need a Handle to dispatch actions.
        The wrapper does NOT own the handle's lifecycle in this case —
        the caller is responsible for shutting it down.
        """
        if agent_name in self._wrappers:
            return self._wrappers[agent_name]

        self._real_handles[agent_name] = real_handle
        wrapper = AcademyAgentHandle(
            agent_name=agent_name,
            manager_wrapper=self,
            real_handle=real_handle,
        )
        self._wrappers[agent_name] = wrapper
        self.logger.info(f"📝 Registered external Academy handle {agent_name!r}")
        return wrapper

    def get_handle(self, agent_name: str) -> "AcademyAgentHandle":
        """Return the ``AcademyAgentHandle`` wrapper for ``agent_name``.

        Creates a wrapper without a real handle if none has been
        registered — callers that then invoke an action on the wrapper
        will get ``AcademyNotImplementedError`` (or a mock response in
        demo mode). This matches the pre-G5 behavior where callers
        could ``get_handle`` before registration.
        """
        if agent_name in self._wrappers:
            return self._wrappers[agent_name]

        self._wrappers[agent_name] = AcademyAgentHandle(
            agent_name=agent_name,
            manager_wrapper=self,
            real_handle=None,  # filled in later by register_agent_class
        )
        self.logger.info(f"🎯 Created placeholder handle for {agent_name!r}")
        return self._wrappers[agent_name]

    async def shutdown(self) -> None:
        """Exit the Manager context and release local state.

        Idempotent. Safe to call multiple times.

        Note: we deliberately rely on Academy's own
        ``Manager.__aexit__`` to shut down every launched agent — it
        iterates ``self.handles`` internally and closes each. An
        explicit per-handle ``manager.shutdown(handle, blocking=True)``
        loop here would race with the ``__aexit__`` cleanup and emit
        ``ExchangeClientNotFoundError`` log-noise on every teardown.
        """
        if self._manager_cm is None:
            self._clear_state()
            return

        try:
            await self._manager_cm.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            # Teardown may race with a failing test's own cleanup.
            # Log at debug — not warning — since we cannot do better
            # once the Manager's context has been partially torn down
            # by an upstream failure.
            self.logger.debug(
                f"Manager __aexit__ raised during shutdown (expected during "
                f"post-failure teardown): {exc}"
            )

        if self._executor is not None:
            self._executor.shutdown(wait=False)

        self._clear_state()
        self.logger.info("✅ Academy Manager shut down cleanly")

    def _clear_state(self) -> None:
        """Reset instance state so a subsequent ``_ensure_manager`` can
        re-initialize cleanly (tests re-use the same wrapper across
        sessions)."""
        self._manager = None
        self._manager_cm = None
        self._launched_handles.clear()
        self._real_handles.clear()
        self._wrappers.clear()
        self._executor = None

    # Backwards-compatible shim: some code may still call the old
    # ``register_agent(agent_name, agent_instance)`` that accepted an
    # ad-hoc instance. Academy does not support registering pre-built
    # instances — the Manager owns lifecycle. Raise with a clear
    # migration hint rather than silently no-op.
    def register_agent(self, agent_name: str, agent_instance: Any) -> None:  # noqa: ARG002
        raise AcademyNotImplementedError(
            "AcademyManagerWrapper.register_agent(agent_name, agent_instance) "
            "is not supported: Academy's Manager owns agent lifecycle. Use "
            "register_agent_class(agent_name, agent_class) to launch + "
            "register in one call, or register_agent_handle(agent_name, "
            "handle) if the agent was launched elsewhere."
        )


class AcademyIntegration:
    """Static utility surface used by workflow loaders and tests."""

    @staticmethod
    def requires_academy_integration(config_path: Union[str, Path]) -> bool:
        """Return True iff the workflow config at ``config_path`` references
        an Academy link."""
        try:
            import yaml

            with open(config_path) as fh:
                config = yaml.safe_load(fh)

            links = config.get("links", {})
            for link_name, link_config in links.items():
                if not isinstance(link_config, dict):
                    continue
                link_class = link_config.get("class", "")
                if "academy_link" in link_class.lower() or "academylink" in link_class:
                    logger.info(
                        f"✅ Found Academy link: {link_name} (class: {link_class})"
                    )
                    return True

                nested = link_config.get("config")
                if isinstance(nested, str) and nested.endswith(".yml"):
                    try:
                        nested_path = Path(config_path).parent / nested
                        with open(nested_path) as nf:
                            nested_data = yaml.safe_load(nf)
                        if (
                            nested_data.get("link_type") == "academy"
                            or "academy_agent_handle" in nested_data
                        ):
                            logger.info(
                                f"✅ Found Academy link in nested config: {nested}"
                            )
                            return True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"⚠️ Could not load nested config {nested}: {exc}"
                        )

            logger.info("ℹ️ No Academy links found in workflow configuration")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"⚠️ Could not check Academy integration requirement: {exc}"
            )
            return False

    @staticmethod
    def setup_academy_manager() -> "AcademyManagerWrapper":
        """Create (or return existing) ``AcademyManagerWrapper`` singleton.

        The real Academy ``Manager`` is not entered here — it enters
        lazily on first ``_ensure_manager`` call. This preserves a
        property of the pre-G5 API: ``setup_academy_manager`` is
        safe to call from non-async code and does not require an event
        loop.

        Raises ``ImportError`` with a clear installation hint when the
        ``academy`` package is not on the path.
        """
        global _GLOBAL_ACADEMY_MANAGER

        if _GLOBAL_ACADEMY_MANAGER is not None:
            logger.info("🔄 Using existing Academy manager singleton")
            return _GLOBAL_ACADEMY_MANAGER

        try:
            from academy.manager import Manager  # noqa: F401
            from academy.exchange import LocalExchangeFactory  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "❌ ACADEMY FRAMEWORK NOT AVAILABLE: "
                f"{exc}\n"
                "   Academy integration is required for this workflow "
                "but the ``academy-py`` package is not installed.\n"
                "   SOLUTION: ``pip install academy-py`` (or install the "
                "project extra that pulls it in).\n"
                "   ALTERNATIVE: Remove Academy links from workflow config."
            ) from exc

        wrapper = AcademyManagerWrapper()
        _GLOBAL_ACADEMY_MANAGER = wrapper
        logger.info("✅ Academy Manager singleton created (lazy lifecycle)")
        return wrapper

    @staticmethod
    def get_academy_manager() -> Optional["AcademyManagerWrapper"]:
        return _GLOBAL_ACADEMY_MANAGER


async def shutdown_academy_manager() -> None:
    """Shut down + clear the process-level singleton.

    Call in test teardown. Idempotent.
    """
    global _GLOBAL_ACADEMY_MANAGER
    if _GLOBAL_ACADEMY_MANAGER is None:
        return
    await _GLOBAL_ACADEMY_MANAGER.shutdown()
    _GLOBAL_ACADEMY_MANAGER = None


__all__ = [
    "AcademyAgentHandle",
    "AcademyIntegration",
    "AcademyManagerWrapper",
    "AcademyNotImplementedError",
    "shutdown_academy_manager",
]
