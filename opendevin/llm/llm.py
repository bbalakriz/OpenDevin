import asyncio
import copy
import warnings
from functools import partial

from opendevin.core.config import LLMConfig
from opendevin.core.message import Message
from opendevin.core.metrics import Metrics
from opendevin.memory.condenser import CondenserMixin

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    ContentPolicyViolationError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from opendevin.core.exceptions import (
    ContextWindowLimitExceededError,
    TokenLimitExceededError,
    UserCancelledError,
)
from opendevin.core.logger import llm_prompt_logger, llm_response_logger
from opendevin.core.logger import opendevin_logger as logger

__all__ = ['LLM', 'BaseLLM']

LOG_MESSAGE_SEPARATOR = '\n\n----------\n\n'
MAX_TOKEN_COUNT_PADDING = 512


class DebugMixin:
    def __init__(self, metrics: Metrics | None = None):
        self.cost_metric_supported = True
        self.metrics = metrics if metrics is not None else Metrics()

    def _log_debug_prompt(self, messages):
        debug_message = self._build_debug_message(messages)
        llm_prompt_logger.debug(debug_message)
        return debug_message

    def _log_debug_response(self, resp, is_streaming=False):
        if is_streaming:
            message_back = resp['choices'][0]['delta']['content']
        else:
            message_back = resp['choices'][0]['message']['content']
        llm_response_logger.debug(message_back)
        self._post_completion(resp)

    def _build_debug_message(self, messages):
        debug_message = ''
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                for element in content:
                    content_str = self._get_content_str(element)
                    debug_message += LOG_MESSAGE_SEPARATOR + content_str
            else:
                content_str = str(content)
                debug_message += LOG_MESSAGE_SEPARATOR + content_str
        return debug_message

    def _get_content_str(self, element):
        if isinstance(element, dict):
            if 'text' in element:
                return element['text'].strip()
            elif 'image_url' in element and 'url' in element['image_url']:
                return element['image_url']['url']
        return str(element)

    def _post_completion(self, response):
        try:
            cur_cost = self.completion_cost(response)
        except Exception:
            cur_cost = 0
        if self.cost_metric_supported:
            logger.info(
                'Cost: %.2f USD | Accumulated Cost: %.2f USD',
                cur_cost,
                self.metrics.accumulated_cost,
            )

    def completion_cost(self, response):
        """Calculate the cost of a completion response based on the model.  Local models are treated as free.
        Add the current cost into total cost in metrics.

        Args:
            response: A response from a model invocation.

        Returns:
            number: The cost of the response.
        """
        if not self.cost_metric_supported:
            return 0.0

        extra_kwargs = {}
        if (
            self.config.input_cost_per_token is not None  # type: ignore
            and self.config.output_cost_per_token is not None  # type: ignore
        ):
            cost_per_token = CostPerToken(
                input_cost_per_token=self.config.input_cost_per_token,  # type: ignore
                output_cost_per_token=self.config.output_cost_per_token,  # type: ignore
            )
            logger.info(f'Using custom cost per token: {cost_per_token}')
            extra_kwargs['custom_cost_per_token'] = cost_per_token

        if not self.is_local():  # type: ignore
            try:
                cost = litellm_completion_cost(
                    completion_response=response, **extra_kwargs
                )
                self.metrics.add_cost(cost)
                return cost
            except Exception:
                self.cost_metric_supported = False
                logger.warning('Cost calculation not supported for this model.')
        return 0.0


class BaseLLM(DebugMixin):
    """The base class for a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
    ):
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Args:
            config: The LLM configuration
        """
        super().__init__(metrics=metrics)
        self.config = copy.deepcopy(config)

        # Set up config attributes with default values to prevent AttributeError
        LLMConfig.set_missing_attributes(self.config)

        # litellm actually uses base Exception here for unknown model
        self.model_info = None
        try:
            if self.config.model.startswith('openrouter'):
                self.model_info = litellm.get_model_info(self.config.model)
            else:
                self.model_info = litellm.get_model_info(
                    self.config.model.split(':')[0]
                )
        # noinspection PyBroadException
        except Exception as e:
            logger.warning(f'Could not get model info for {config.model}:\n{e}')

        # Set the max tokens in an LM-specific way if not set
        if self.config.max_input_tokens is None:
            if (
                self.model_info is not None
                and 'max_input_tokens' in self.model_info
                and isinstance(self.model_info['max_input_tokens'], int)
            ):
                self.config.max_input_tokens = self.model_info['max_input_tokens']
            else:
                # Max input tokens for gpt3.5, so this is a safe fallback for any potentially viable model
                self.config.max_input_tokens = 4096

        if self.config.max_output_tokens is None:
            if (
                self.model_info is not None
                and 'max_output_tokens' in self.model_info
                and isinstance(self.model_info['max_output_tokens'], int)
            ):
                self.config.max_output_tokens = self.model_info['max_output_tokens']
            else:
                # Max output tokens for gpt3.5, so this is a safe fallback for any potentially viable model
                self.config.max_output_tokens = 1024

        if self.config.drop_params:
            litellm.drop_params = self.config.drop_params

        self._completion = partial(
            litellm_completion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        completion_unwrapped = self._completion

        # Define async_completion_unwrapped using self._call_acompletion
        async_completion_unwrapped = partial(
            self._call_acompletion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=True,
        )

        self._completion = self._create_sync_wrapper(completion_unwrapped)
        self._async_completion = self._create_async_wrapper(async_completion_unwrapped)
        self._async_streaming_completion = self._create_async_streaming_wrapper(
            async_completion_unwrapped
        )

    def _create_sync_wrapper(self, completion_unwrapped):
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=self._attempt_on_error,
        )
        def retry_wrapper(*args, **kwargs):
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            debug_message = self._log_debug_prompt(messages)

            # skip if messages is empty (thus debug_message is empty)
            if debug_message:
                resp = completion_unwrapped(*args, **kwargs)
                self._log_debug_response(resp)
            else:
                resp = {'choices': [{'message': {'content': ''}}]}

            return resp

        return retry_wrapper

    def _create_async_wrapper(self, async_completion_unwrapped):
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=self._attempt_on_error,
        )
        async def async_completion_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            debug_message = self._log_debug_prompt(messages)

            async def check_stopped():
                while True:
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError('LLM request cancelled by user')
                    await asyncio.sleep(0.1)

            stop_check_task = asyncio.create_task(check_stopped())

            try:
                # skip if messages is empty (thus debug_message is empty)
                if debug_message:
                    resp = await async_completion_unwrapped(*args, **kwargs)
                    self._log_debug_response(resp)
                else:
                    resp = {'choices': [{'message': {'content': ''}}]}
                return resp

            except UserCancelledError:
                logger.info('LLM request cancelled by user.')
                raise
            except OpenAIError as e:
                logger.error(f'OpenAIError occurred:\n{e}')
                raise
            except (
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
                InternalServerError,
            ) as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise
            finally:
                await asyncio.sleep(0.1)
                stop_check_task.cancel()
                try:
                    await stop_check_task
                except asyncio.CancelledError:
                    pass

        return async_completion_wrapper

    def _create_async_streaming_wrapper(self, async_completion_unwrapped):
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=self._attempt_on_error,
        )
        async def async_acompletion_stream_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion with streaming function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            self._log_debug_prompt(messages)

            try:
                # Directly call and await litellm_acompletion
                resp = await async_completion_unwrapped(*args, **kwargs)

                # For streaming we iterate over the chunks
                async for chunk in resp:
                    # Check for cancellation before yielding the chunk
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError(
                            'LLM request cancelled due to CANCELLED state'
                        )
                    self._log_debug_response(chunk, is_streaming=True)
                    yield chunk

            except UserCancelledError:
                logger.info('LLM request cancelled by user.')
                raise
            except OpenAIError as e:
                logger.error(f'OpenAIError occurred:\n{e}')
                raise
            except (
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
                InternalServerError,
            ) as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise
            finally:
                if kwargs.get('stream', False):
                    await asyncio.sleep(0.1)

        return async_acompletion_stream_wrapper

    def _attempt_on_error(self, retry_state):
        logger.error(
            f'{retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number} | You can customize these settings in the configuration.',
            exc_info=False,
        )
        return None

    async def _call_acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **kwargs)

    @property
    def completion(self):
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion

    @property
    def async_completion(self):
        """Decorator for the async litellm acompletion function.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        return self._async_completion

    @property
    def async_streaming_completion(self):
        """Decorator for the async litellm acompletion function with streaming.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        return self._async_streaming_completion

    def supports_vision(self):
        return litellm.supports_vision(self.config.model)

    def is_over_token_limit(self, messages: list[Message]) -> bool:
        """
        Estimates the token count of the given events using litellm tokenizer and returns True if over the max_input_tokens value.

        Parameters:
        - messages: List of messages to estimate the token count for.

        Returns:
        - Boolean indicating whether the token count is over the limit.
        """
        # max_input_tokens will always be set in init to some sensible default
        # 0 in config.llm disables the check
        if not self.config.max_input_tokens:
            return False
        token_count = (
            litellm.token_counter(model=self.config.model, messages=messages)
            + MAX_TOKEN_COUNT_PADDING
        )
        return token_count >= self.config.max_input_tokens

    def get_text_messages(self, messages: list[Message]) -> list[dict]:
        text_messages = []
        for message in messages:
            text_messages.append(message.message)
        return text_messages

    def is_local(self):
        """Determines if the system is using a locally running LLM.

        Returns:
            boolean: True if executing a local model.
        """
        if self.config.base_url is not None:
            for substring in ['localhost', '127.0.0.1' '0.0.0.0']:
                if substring in self.config.base_url:
                    return True
        elif self.config.model is not None:
            if self.config.model.startswith('ollama'):
                return True
        return False

    def __str__(self):
        if self.config.api_version:
            return f'LLM(model={self.config.model}, api_version={self.config.api_version}, base_url={self.config.base_url})'
        elif self.config.base_url:
            return f'LLM(model={self.config.model}, base_url={self.config.base_url})'
        return f'LLM(model={self.config.model})'

    def __repr__(self):
        return str(self)

    def reset(self):
        self.metrics = Metrics()


class LLM(BaseLLM, CondenserMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(self, config: LLMConfig, metrics: Metrics | None = None):
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Args:
            config: The LLM configuration
        """
        super().__init__(config, metrics)

    def _create_sync_wrapper(self, completion_unwrapped):
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=self._attempt_on_error,
        )
        def retry_wrapper(*args, **kwargs):
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            debug_message = self._log_debug_prompt(messages)

            try:
                if self.is_over_token_limit(messages):
                    raise TokenLimitExceededError()
            except TokenLimitExceededError:
                # If we got a token limit exceeded error, try condensing the messages, then try again
                if kwargs['condense'] and self.is_over_token_limit(messages):
                    # A separate call to run a summarizer
                    summary_action = self.condense(messages=messages)
                    return summary_action
                else:
                    logger.debug('step() failed with an unrecognized exception:')
                    raise ContextWindowLimitExceededError()

            # if we get here, the token limit has not been exceeded
            # get the messages in the form of list(str)
            # text_messages = self.get_text_messages(messages)

            # call the completion function
            # skip if messages is empty (thus debug_message is empty)
            if debug_message:
                resp = completion_unwrapped(*args, **kwargs)
                self._log_debug_response(resp)
            else:
                resp = {'choices': [{'message': {'content': ''}}]}

            return resp

        return retry_wrapper
