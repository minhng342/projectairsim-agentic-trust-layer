"""No test may call a real LLM by accident (deck slide 9): model requests are blocked for the
whole suite. Offline models (TestModel, FunctionModel, the scripted model) are unaffected."""
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False
