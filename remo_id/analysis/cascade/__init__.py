"""N-stage cascade analysis.

Consumes multiple (checkpoint, class_config) pairs whose predictions have
already been saved, auto-infers the decision tree from the class_configs,
and produces end-to-end cascade performance metrics and figures.
"""

from remo_id.analysis.cascade.report import run_cascade

__all__ = ["run_cascade"]
