"""Trusted flight-safety infrastructure for team runs (docs/recovery_policy.md).

The transition manager stands in for a UTM or vertiport service. It reserves vertical
moves so that drones changing altitude stay separated. It is explicitly trusted: it does
not allocate search sectors, change beliefs, or give the agents simulator truth, and the
same manager runs in every comparison condition. Its separation results are not evidence
that the agents' trust layer keeps drones apart.
"""
