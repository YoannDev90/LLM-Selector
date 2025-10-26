import logging

logger_name = "llm_selector"

LLM_SELECTOR_PROMPT = """You are an intelligent LLM selector. Based on the user's query, select the most appropriate LLM model from the list below.
Consider factors like complexity, speed requirements, and task type.
Models with higher weights should be preferred proportionally to their weight value.
Respond with only the model name, nothing else.

Available models:
"""
