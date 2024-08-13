# Memory Component

- Short Term History
- Memory Condenser
- Long Term Memory

## Short Term History
- Short term history filters the event stream and computes the messages that are injected into the context
- It filters out certain events of no interest for the Agent, such as AgentChangeStateObservation or NullAction/NullObservation
- When the context window or the token limit set by the user is exceeded, history starts condensing: chunks of messages into summaries.
- Each summary is then injected into the context, in the place of the respective chunk it summarizes

## Memory Condenser
- Memory condenser is responsible for summarizing the chunks of events
- It summarizes the earlier events first
- It starts with the earliest agent actions and observations between two user messages
- Then it does the same for later chunks of events between user messages
- If there are no more agent events, it summarizes the user messages, this time one by one, if they're large enough and not immediately after an AgentFinishAction event (we assume those are tasks, potentially important)
- Summaries are retrieved from the LLM as AgentSummarizeAction, and are saved in State.

## Long Term Memory
- Long term memory component stores embeddings for events and prompts in a vector store
- The agent can query it when it needs detailed information about a past event or to learn new actions


```mermaid
classDiagram
    class Agent {
        +ContextWindow
        +step()
    }
    class Memory {
        +insertIntoContext()
    }
    class History {
        +get_events()
        +on_event()
        +get_last_user_message()
    }
    class LongTermMemory {
        +store()
        +retrieve()
    }
    class MemoryCondenser {
        +summarize()
    }
    class AgentSummarizeAction {
    }
    class AgentRecallAction {
    }

    Agent "1" -- "1" Memory : has
    Memory "1" -- "1" History : includes
    Memory "1" -- "1" LongTermMemory : includes
    Memory "1" -- "1" MemoryCondenser : includes
    Agent "1" -- "*" AgentSummarizeAction : performs
    Agent "1" -- "*" AgentRecallAction : performs

    note for History "short term memory"
    note for Memory "Components insert into ContextWindow"
```​​​​​​​​​​​​​​​​
