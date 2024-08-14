from opendevin.controller.agent import Agent

from .mem_codeact_agent import MemCodeActAgent

Agent.register('MemCodeActAgent', MemCodeActAgent)
