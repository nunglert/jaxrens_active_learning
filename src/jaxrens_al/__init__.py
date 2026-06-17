"""jaxrens_active_learning -- composition root for jaxrens AL projects.

Ties together the layered packages:
    * ``remote_handler`` -- orchestration (loop, ABCs, evaluators, trainers)
    * ``alviz``          -- dashboard engine + generic panels
    * (future) ``sampler`` -- the pure sampling method

This package owns the application-specific glue: the dashboard assembly that
reads a run's artifacts and feeds alviz (:func:`jaxrens_al.dashboard.
build_dashboard`), project-specific panels, and the loop-wiring examples.
"""

from jaxrens_al.templates import template_dir

# ``build_dashboard`` is exposed lazily (PEP 562): importing it eagerly would
# pull in the whole panels stack + ``alviz`` for *every* ``import jaxrens_al``,
# coupling lightweight, dashboard-free tools (e.g. ``jaxrens_al.reference_db``,
# which is pure numpy + ase) to the dashboard engine. ``jaxrens_al.
# build_dashboard`` still works -- it just imports on first access. Same reason
# the concrete ``JaxrensNSSampler`` is not imported here: it would pull in
# ``remote_handler`` (the [run] extra); import it via ``jaxrens_al.sampler``.
__all__ = ["build_dashboard", "template_dir"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "build_dashboard":
        from jaxrens_al.dashboard import build_dashboard

        return build_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
