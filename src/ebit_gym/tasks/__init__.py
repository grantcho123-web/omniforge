"""Task-type helpers.

Tasks are just data — JSON-serializable instances of ``Task``. This
subpackage holds factories that produce well-formed ``Task`` objects
for specific domains, plus any grader/runner glue those domains need
that doesn't fit the general-purpose schema.

Currently here:

* ``trading`` — wraps the v0.1 single-asset env as a scoring task. The
  Gymnasium env itself stays put under ``ebit_gym.envs`` for users who
  want sequential RL.
"""

from ebit_gym.tasks.trading import (
    SimulatedTradingTaskBuilder,
    score_position,
)

__all__ = ["SimulatedTradingTaskBuilder", "score_position"]
