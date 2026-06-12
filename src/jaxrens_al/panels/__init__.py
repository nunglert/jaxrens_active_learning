"""Project-specific dashboard panels.

These subclass ``alviz.Entry`` but may depend on this project's knowledge of
remote_handler's run layout (or, in future, the sampler's data formats).
Generic, reusable panels belong in ``alviz.panels`` instead.
"""

from jaxrens_al.panels.progress import DatasetGrowthPanel

__all__ = ["DatasetGrowthPanel"]
