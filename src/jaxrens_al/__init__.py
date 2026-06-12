"""jaxrens_active_learning -- composition root for jaxrens AL projects.

Ties together the layered packages:
    * ``remote_handler`` -- orchestration (loop, ABCs, evaluators, trainers)
    * ``alviz``          -- dashboard engine + generic panels
    * (future) ``sampler`` -- the pure sampling method

This package owns the application-specific glue: the dashboard assembly that
reads a run's artifacts and feeds alviz (:func:`jaxrens_al.dashboard.
build_dashboard`), project-specific panels, and the loop-wiring examples.
"""

from jaxrens_al.dashboard import build_dashboard

__all__ = ["build_dashboard"]
__version__ = "0.1.0"
