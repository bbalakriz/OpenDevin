from opendevin.core.exceptions import (
    SummarizeError,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.message import Message
from opendevin.events.action.agent import (
    AgentDelegateSummaryAction,
    AgentSummarizeAction,
)
from opendevin.events.event import Event
from opendevin.events.serialization.event import event_to_memory
from opendevin.llm.llm import LLM
from opendevin.memory.prompts import (
    get_delegate_summarize_prompt,
    get_summarize_prompt,
    parse_delegate_summary_response,
    parse_summary_response,
)

MAX_USER_MESSAGE_CHAR_COUNT = 200  # max char count for user messages
MESSAGE_SUMMARY_WARNING_FRAC = 0.75


class MemoryCondenser:
    llm: LLM

    def __init__(self, llm: LLM):
        self.llm = llm

    def condense(self, summarize_prompt: str, llm: LLM):
        """Attempts to condense the memory by using the llm

        Parameters:
        - llm (LLM): llm to be used for summarization

        Raises:
        - Exception: the same exception as it got from the llm or processing the response
        """
        try:
            messages = [{'content': summarize_prompt, 'role': 'user'}]
            resp = llm.completion(messages=messages)
            summary_response = resp['choices'][0]['message']['content']
            return summary_response
        except Exception as e:
            logger.error('Error condensing thoughts: %s', str(e), exc_info=False)

            # TODO If the llm fails with ContextWindowExceededError, we can try to condense the memory chunk by chunk
            raise

    def summarize_delegate(
        self, delegate_events: list[Event], delegate_agent: str, delegate_task: str
    ) -> AgentDelegateSummaryAction:
        """
        Summarizes the given list of events into a concise summary.

        Parameters:
        - delegate_events: List of events of the delegate.
        - delegate_agent: The agent that was delegated to.
        - delegate_task: The task that was delegated.

        Returns:
        - The summary of the delegate's activities.
        """
        try:
            event_dicts = [event_to_memory(event, 10_000) for event in delegate_events]
            prompt = get_delegate_summarize_prompt(
                event_dicts, delegate_agent, delegate_task
            )

            messages = [{'role': 'user', 'content': prompt}]
            response = self.llm.completion(messages=messages)

            action_response: str = response['choices'][0]['message']['content']
            action = parse_delegate_summary_response(action_response)
            action.task = delegate_task
            action.agent = delegate_agent
            return action
        except Exception as e:
            logger.error(f'Failed to summarize delegate events: {e}')
            raise


class CondenserMixin:
    """Condenses a group of condensable messages as done by MemGPT."""

    def condense(
        self,
        messages: list[Message],
    ):
        # Start past the system message, and example messages.,
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation  for in-context examples of function calling
        token_counts = [
            self.get_token_count([message])  # type: ignore
            for message in messages
            if message.condensable
        ]
        message_buffer_token_count = sum(token_counts)  # no system and example message

        desired_token_count_to_summarize = int(
            message_buffer_token_count * self.config.message_summary_trunc_tokens_frac  # type: ignore
        )

        candidate_messages_to_summarize = []
        tokens_so_far = 0
        for message in messages:
            if message.condensable:
                candidate_messages_to_summarize.append(message)
                tokens_so_far += self.get_token_count([message])  # type: ignore
            if tokens_so_far > desired_token_count_to_summarize:
                last_summarized_event_id = message.event_id
                break

        # TODO: Add functionality for preserving last N messages
        # MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST = 3
        # if preserve_last_N_messages:
        #     candidate_messages_to_summarize = candidate_messages_to_summarize[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
        #     token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]

        logger.debug(
            f'message_summary_trunc_tokens_frac={self.config.message_summary_trunc_tokens_frac}'  # type: ignore
        )
        # logger.debug(f'MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}')
        logger.debug(f'token_counts={token_counts}')
        logger.debug(f'message_buffer_token_count={message_buffer_token_count}')
        logger.debug(
            f'desired_token_count_to_summarize={desired_token_count_to_summarize}'
        )
        logger.debug(
            f'len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}'
        )

        if len(candidate_messages_to_summarize) == 0:
            raise SummarizeError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(messages)}]"
            )

        # TODO: Try to make an assistant message come after the cutoff

        message_sequence_to_summarize = candidate_messages_to_summarize

        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise SummarizeError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            print(
                f'Attempting to summarize with last summarized event id = {last_summarized_event_id}'
            )

        action_response = self.summarize_messages(
            message_sequence_to_summarize=message_sequence_to_summarize
        )
        summary_action: AgentSummarizeAction = parse_summary_response(action_response)
        summary_action.last_summarized_event_id = (
            last_summarized_event_id if last_summarized_event_id else -1
        )
        return summary_action

    def summarize_messages(self, message_sequence_to_summarize: list[Message]):
        """Summarize a message sequence using LLM"""
        context_window = self.config.max_input_tokens  # type: ignore
        summary_prompt = get_summarize_prompt(
            self.get_text_messages(message_sequence_to_summarize)  # type: ignore
        )
        summary_input_tkns = self.get_token_count(summary_prompt)  # type: ignore
        if context_window is None:
            raise ValueError('context_window should not be None')
        if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
            trunc_ratio = (
                MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns
            ) * 0.8  # For good measure...
            cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
            curr_summary = self.summarize_messages(
                message_sequence_to_summarize=message_sequence_to_summarize[:cutoff]
            )
            curr_summary_message = (
                'Summary of all Action and Observations till now. \n'
                + 'Action: '
                + curr_summary['args']['summarized_actions']
                + '\nObservation: '
                + curr_summary['args']['summarized_observations']
            )
            input = [
                Message({'role': 'assistant', 'content': curr_summary_message})
            ] + message_sequence_to_summarize[cutoff:]
            summary_input = self._format_summary_history(self.get_text_messages(input))  # type: ignore

        message_sequence = []
        message_sequence.append(Message({'role': 'system', 'content': summary_prompt}))
        message_sequence.append(Message({'role': 'user', 'content': summary_input}))

        response = self.completion(  # type: ignore
            messages=message_sequence,
            stop=[
                '</execute_ipython>',
                '</execute_bash>',
                '</execute_browse>',
            ],
            temperature=0.0,
        )

        print(f'summarize_messages got reply: {response.choices[0]}')

        action_response = response['choices'][0]['message']['content']
        return action_response
