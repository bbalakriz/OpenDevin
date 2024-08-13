from dataclasses import dataclass, field

from opendevin.core.schema import ActionType

from .action import Action


@dataclass
class ChangeAgentStateAction(Action):
    """Fake action, just to notify the client that a task state has changed."""

    agent_state: str
    thought: str = ''
    action: str = ActionType.CHANGE_AGENT_STATE

    @property
    def message(self) -> str:
        return f'Agent state changed to {self.agent_state}'


@dataclass
class AgentSummarizeAction(Action):
    summarized_actions: str = ''
    summarized_observations: str = ''
    action: str = ActionType.SUMMARIZE
    last_summarized_event_id: int = -1

    @property
    def message(self) -> str:
        return self.summarized_actions + '\n' + self.summarized_observations

    def __str__(self) -> str:
        ret = '**AgentSummarizeAction**\n'
        ret += f'SUMMARIZED ACTIONS: {self.summarized_actions}\n'
        ret += f'SUMMARIZED OBSERVATIONS: {self.summarized_observations}'
        return ret


@dataclass
class AgentDelegateSummaryAction(Action):
    """
    Action to summarize a list of events.

    Attributes:
    - agent: What agent was delegated to.
    - task: The task that was delegated.
    - summary: A single sentence summarizing all the delegate's actions and observations.
    """

    agent: str = ''
    task: str = ''
    summary: str = ''
    relevant_info: str = ''
    action: str = ActionType.SUMMARIZE_DELEGATE
    _chunk_start: int = -1
    _chunk_end: int = -1

    @property
    def message(self) -> str:
        return self.summary

    def __str__(self) -> str:
        ret = '**AgentDelegateSummaryAction**\n'
        ret += f'SUMMARY: {self.summary}'
        return ret


@dataclass
class AgentFinishAction(Action):
    outputs: dict = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.FINISH

    @property
    def message(self) -> str:
        if self.thought != '':
            return self.thought
        return "All done! What's next on the agenda?"


@dataclass
class AgentRejectAction(Action):
    outputs: dict = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.REJECT

    @property
    def message(self) -> str:
        msg: str = 'Task is rejected by the agent.'
        if 'reason' in self.outputs:
            msg += ' Reason: ' + self.outputs['reason']
        return msg


@dataclass
class AgentDelegateAction(Action):
    agent: str
    inputs: dict
    thought: str = ''
    action: str = ActionType.DELEGATE

    @property
    def message(self) -> str:
        return f"I'm asking {self.agent} for help with this task."
