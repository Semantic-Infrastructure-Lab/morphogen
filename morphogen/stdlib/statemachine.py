"""
State Machine Domain

Provides state machine and behavior modeling functionality:
- Finite State Machines (FSM)
- Hierarchical State Machines
- Behavior trees
- State transitions with guards and actions
- Event-driven state management

Useful for game AI, UI flows, protocol implementations, and workflow systems.

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from enum import Enum
from collections import defaultdict

from morphogen.core.operator import operator, OpCategory


class TransitionType(Enum):
    """Transition types"""
    AUTOMATIC = "automatic"  # Transition happens automatically if guard passes
    EVENT = "event"          # Transition requires specific event
    TIMEOUT = "timeout"      # Transition after time delay


@dataclass
class State:
    """State in a finite state machine

    Attributes:
        name: State name/identifier
        on_enter: Optional callback when entering state
        on_exit: Optional callback when exiting state
        on_update: Optional callback each update while in state
        data: Optional state-specific data
    """
    name: str
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    on_update: Optional[Callable] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        return False


@dataclass
class Transition:
    """Transition between states

    Attributes:
        from_state: Source state
        to_state: Target state
        event: Event name that triggers transition (None for automatic)
        guard: Optional condition function that must return True
        action: Optional function to execute during transition
        priority: Priority for resolving multiple valid transitions (higher = first)
        transition_type: Type of transition
        timeout: Timeout duration in seconds (for TIMEOUT type)
    """
    from_state: str
    to_state: str
    event: Optional[str] = None
    guard: Optional[Callable] = None
    action: Optional[Callable] = None
    priority: int = 0
    transition_type: TransitionType = TransitionType.EVENT
    timeout: float = 0.0


@dataclass
class StateMachine:
    """Finite State Machine

    Attributes:
        name: Machine name
        states: Dict of state name -> State
        transitions: List of transitions
        current_state: Current state name
        initial_state: Initial state name
        context: Shared context data accessible to all states
        time_in_state: Time spent in current state (seconds)
    """
    name: str
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    current_state: Optional[str] = None
    initial_state: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    time_in_state: float = 0.0

    def copy(self) -> 'StateMachine':
        """Create a copy of the state machine"""
        return StateMachine(
            name=self.name,
            states=dict(self.states),
            transitions=list(self.transitions),
            current_state=self.current_state,
            initial_state=self.initial_state,
            context=dict(self.context),
            time_in_state=self.time_in_state
        )


@dataclass
class BehaviorNode:
    """Node in a behavior tree

    Attributes:
        name: Node name
        node_type: 'sequence', 'selector', 'action', 'condition', 'parallel'
        children: Child nodes
        action: Action function for action nodes
        condition: Condition function for condition nodes
        data: Node-specific data
    """
    name: str
    node_type: str
    children: List['BehaviorNode'] = field(default_factory=list)
    action: Optional[Callable] = None
    condition: Optional[Callable] = None
    data: Dict[str, Any] = field(default_factory=dict)


class BehaviorStatus(Enum):
    """Behavior execution status"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class StateMachineOperations:
    """State machine operations"""

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.CONSTRUCT,
        signature="(name: str, initial_state: str) -> StateMachine",
        deterministic=True,
        doc="Create a new state machine"
    )
    def create(name: str, initial_state: str) -> StateMachine:
        """Create a new state machine

        Args:
            name: Machine name
            initial_state: Initial state name

        Returns:
            New state machine
        """
        return StateMachine(
            name=name,
            initial_state=initial_state,
            current_state=initial_state
        )

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(fsm: StateMachine, state: State) -> StateMachine",
        deterministic=True,
        doc="Add a state to the machine"
    )
    def add_state(fsm: StateMachine, state: State) -> StateMachine:
        """Add a state to the machine

        Args:
            fsm: State machine
            state: State to add

        Returns:
            Updated state machine
        """
        result = fsm.copy()
        result.states[state.name] = state
        return result

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(fsm: StateMachine, transition: Transition) -> StateMachine",
        deterministic=True,
        doc="Add a transition to the machine"
    )
    def add_transition(fsm: StateMachine, transition: Transition) -> StateMachine:
        """Add a transition to the machine

        Args:
            fsm: State machine
            transition: Transition to add

        Returns:
            Updated state machine
        """
        result = fsm.copy()
        result.transitions.append(transition)
        return result

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(fsm: StateMachine) -> StateMachine",
        deterministic=False,  # May call user callbacks
        doc="Start the state machine (enter initial state)"
    )
    def start(fsm: StateMachine) -> StateMachine:
        """Start the state machine (enter initial state)

        Args:
            fsm: State machine

        Returns:
            Started state machine
        """
        result = fsm.copy()
        result.current_state = result.initial_state
        result.time_in_state = 0.0

        # Call on_enter for initial state
        if result.current_state and result.current_state in result.states:
            state = result.states[result.current_state]
            if state.on_enter:
                state.on_enter(result.context)

        return result

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(fsm: StateMachine, dt: float) -> StateMachine",
        deterministic=False,  # May trigger transitions with callbacks
        doc="Update state machine (check automatic transitions and timeouts)"
    )
    def update(fsm: StateMachine, dt: float) -> StateMachine:
        """Update state machine (check automatic transitions and timeouts)

        Args:
            fsm: State machine
            dt: Time delta in seconds

        Returns:
            Updated state machine
        """
        result = fsm.copy()
        result.time_in_state += dt

        if not result.current_state:
            return result

        # Call on_update for current state
        current = result.states.get(result.current_state)
        if current and current.on_update:
            current.on_update(result.context, dt)

        # Check for timeout transitions
        timeout_transitions = [
            t for t in result.transitions
            if t.from_state == result.current_state
            and t.transition_type == TransitionType.TIMEOUT
            and result.time_in_state >= t.timeout
        ]

        if timeout_transitions:
            # Take highest priority timeout transition
            timeout_transitions.sort(key=lambda t: t.priority, reverse=True)
            trans = timeout_transitions[0]
            return StateMachineOperations._execute_transition(result, trans)

        # Check for automatic transitions
        auto_transitions = [
            t for t in result.transitions
            if t.from_state == result.current_state
            and t.transition_type == TransitionType.AUTOMATIC
        ]

        # Sort by priority
        auto_transitions.sort(key=lambda t: t.priority, reverse=True)

        for trans in auto_transitions:
            # Check guard condition
            if trans.guard is None or trans.guard(result.context):
                return StateMachineOperations._execute_transition(result, trans)

        return result

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(fsm: StateMachine, event: str) -> StateMachine",
        deterministic=False,  # May trigger transitions with callbacks
        doc="Send an event to the state machine"
    )
    def send_event(fsm: StateMachine, event: str) -> StateMachine:
        """Send an event to the state machine

        Args:
            fsm: State machine
            event: Event name

        Returns:
            Updated state machine
        """
        result = fsm.copy()

        if not result.current_state:
            return result

        # Find matching event transitions
        event_transitions = [
            t for t in result.transitions
            if t.from_state == result.current_state
            and t.event == event
            and t.transition_type == TransitionType.EVENT
        ]

        if not event_transitions:
            return result

        # Sort by priority
        event_transitions.sort(key=lambda t: t.priority, reverse=True)

        for trans in event_transitions:
            # Check guard condition
            if trans.guard is None or trans.guard(result.context):
                return StateMachineOperations._execute_transition(result, trans)

        return result

    @staticmethod
    def _execute_transition(fsm: StateMachine, trans: Transition) -> StateMachine:
        """Execute a state transition

        Args:
            fsm: State machine
            trans: Transition to execute

        Returns:
            State machine after transition
        """
        # Call on_exit for current state
        if fsm.current_state in fsm.states:
            current = fsm.states[fsm.current_state]
            if current.on_exit:
                current.on_exit(fsm.context)

        # Execute transition action
        if trans.action:
            trans.action(fsm.context)

        # Change state
        fsm.current_state = trans.to_state
        fsm.time_in_state = 0.0

        # Call on_enter for new state
        if fsm.current_state in fsm.states:
            new_state = fsm.states[fsm.current_state]
            if new_state.on_enter:
                new_state.on_enter(fsm.context)

        return fsm

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.QUERY,
        signature="(fsm: StateMachine) -> Optional[str]",
        deterministic=True,
        doc="Get current state name"
    )
    def get_state_name(fsm: StateMachine) -> Optional[str]:
        """Get current state name

        Args:
            fsm: State machine

        Returns:
            Current state name or None
        """
        return fsm.current_state

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.QUERY,
        signature="(fsm: StateMachine, state_name: str) -> bool",
        deterministic=True,
        doc="Check if machine is in given state"
    )
    def is_in_state(fsm: StateMachine, state_name: str) -> bool:
        """Check if machine is in given state

        Args:
            fsm: State machine
            state_name: State name to check

        Returns:
            True if in state
        """
        return fsm.current_state == state_name

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.QUERY,
        signature="(fsm: StateMachine) -> List[Transition]",
        deterministic=True,
        doc="Get all valid transitions from current state"
    )
    def get_valid_transitions(fsm: StateMachine) -> List[Transition]:
        """Get all valid transitions from current state

        Args:
            fsm: State machine

        Returns:
            List of valid transitions
        """
        if not fsm.current_state:
            return []

        return [
            t for t in fsm.transitions
            if t.from_state == fsm.current_state
        ]

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.QUERY,
        signature="(fsm: StateMachine) -> str",
        deterministic=True,
        doc="Convert state machine to Graphviz DOT format"
    )
    def to_graphviz(fsm: StateMachine) -> str:
        """Convert state machine to Graphviz DOT format

        Args:
            fsm: State machine

        Returns:
            DOT format string
        """
        dot = f'digraph "{fsm.name}" {{\n'
        dot += '  rankdir=LR;\n'
        dot += '  node [shape=circle];\n'

        # Mark initial state
        if fsm.initial_state:
            dot += f'  start [shape=point];\n'
            dot += f'  start -> "{fsm.initial_state}";\n'

        # Mark current state
        if fsm.current_state:
            dot += f'  "{fsm.current_state}" [peripheries=2];\n'

        # Add transitions
        for trans in fsm.transitions:
            label = trans.event or "auto"
            if trans.transition_type == TransitionType.TIMEOUT:
                label = f"timeout({trans.timeout}s)"

            dot += f'  "{trans.from_state}" -> "{trans.to_state}" [label="{label}"];\n'

        dot += '}\n'
        return dot

    # Behavior Tree Operations

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.CONSTRUCT,
        signature="(name: str, children: List[BehaviorNode]) -> BehaviorNode",
        deterministic=True,
        doc="Create a sequence node (succeeds if all children succeed)"
    )
    def create_sequence(name: str, children: List[BehaviorNode]) -> BehaviorNode:
        """Create a sequence node (succeeds if all children succeed)

        Args:
            name: Node name
            children: Child nodes

        Returns:
            Sequence behavior node
        """
        return BehaviorNode(name=name, node_type='sequence', children=children)

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.CONSTRUCT,
        signature="(name: str, children: List[BehaviorNode]) -> BehaviorNode",
        deterministic=True,
        doc="Create a selector node (succeeds if any child succeeds)"
    )
    def create_selector(name: str, children: List[BehaviorNode]) -> BehaviorNode:
        """Create a selector node (succeeds if any child succeeds)

        Args:
            name: Node name
            children: Child nodes

        Returns:
            Selector behavior node
        """
        return BehaviorNode(name=name, node_type='selector', children=children)

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.CONSTRUCT,
        signature="(name: str, action: Callable) -> BehaviorNode",
        deterministic=True,
        doc="Create an action node"
    )
    def create_action(name: str, action: Callable) -> BehaviorNode:
        """Create an action node

        Args:
            name: Node name
            action: Action function (takes context, returns BehaviorStatus)

        Returns:
            Action behavior node
        """
        return BehaviorNode(name=name, node_type='action', action=action)

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.CONSTRUCT,
        signature="(name: str, condition: Callable) -> BehaviorNode",
        deterministic=True,
        doc="Create a condition node"
    )
    def create_condition(name: str, condition: Callable) -> BehaviorNode:
        """Create a condition node

        Args:
            name: Node name
            condition: Condition function (takes context, returns bool)

        Returns:
            Condition behavior node
        """
        return BehaviorNode(name=name, node_type='condition', condition=condition)

    @staticmethod
    def _exec_action(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        if node.action:
            return node.action(context)
        return BehaviorStatus.FAILURE

    @staticmethod
    def _exec_condition(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        if node.condition:
            return BehaviorStatus.SUCCESS if node.condition(context) else BehaviorStatus.FAILURE
        return BehaviorStatus.FAILURE

    @staticmethod
    def _exec_sequence(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        for child in node.children:
            status = StateMachineOperations.execute_behavior(child, context)
            if status != BehaviorStatus.SUCCESS:
                return status
        return BehaviorStatus.SUCCESS

    @staticmethod
    def _exec_selector(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        for child in node.children:
            status = StateMachineOperations.execute_behavior(child, context)
            if status in (BehaviorStatus.SUCCESS, BehaviorStatus.RUNNING):
                return status
        return BehaviorStatus.FAILURE

    @staticmethod
    def _exec_parallel(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        results = [StateMachineOperations.execute_behavior(c, context) for c in node.children]
        if all(s == BehaviorStatus.SUCCESS for s in results):
            return BehaviorStatus.SUCCESS
        if any(s == BehaviorStatus.RUNNING for s in results):
            return BehaviorStatus.RUNNING
        return BehaviorStatus.FAILURE

    @staticmethod
    @operator(
        domain="statemachine",
        category=OpCategory.TRANSFORM,
        signature="(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus",
        deterministic=False,  # May execute user actions
        doc="Execute a behavior tree node"
    )
    def execute_behavior(node: BehaviorNode, context: Dict[str, Any]) -> BehaviorStatus:
        """Execute a behavior tree node."""
        _dispatch = {
            'action': StateMachineOperations._exec_action,
            'condition': StateMachineOperations._exec_condition,
            'sequence': StateMachineOperations._exec_sequence,
            'selector': StateMachineOperations._exec_selector,
            'parallel': StateMachineOperations._exec_parallel,
        }
        handler = _dispatch.get(node.node_type)
        return handler(node, context) if handler else BehaviorStatus.FAILURE


# Export singleton instance for DSL access
statemachine = StateMachineOperations()

# Export all operators for registry discovery
# FSM operators
create = StateMachineOperations.create
add_state = StateMachineOperations.add_state
add_transition = StateMachineOperations.add_transition
start = StateMachineOperations.start
update = StateMachineOperations.update
send_event = StateMachineOperations.send_event
get_state_name = StateMachineOperations.get_state_name
is_in_state = StateMachineOperations.is_in_state
get_valid_transitions = StateMachineOperations.get_valid_transitions
to_graphviz = StateMachineOperations.to_graphviz

# Behavior tree operators
create_sequence = StateMachineOperations.create_sequence
create_selector = StateMachineOperations.create_selector
create_action = StateMachineOperations.create_action
create_condition = StateMachineOperations.create_condition
execute_behavior = StateMachineOperations.execute_behavior
