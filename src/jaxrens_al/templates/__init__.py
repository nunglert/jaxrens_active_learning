"""SLURM job-body templates shipped by jaxrens_al.

``JaxrensNSSampler`` builds its SLURM job through ``remote_handler``'s
templating, which renders a cluster *header* template and ``{% include %}``s a
*body* template found on ``remote_handler``'s template search path. The body
that runs ``jaxrens run`` (``body_jaxrens.sh.j2``) lives here rather than in
``remote_handler`` -- the orchestrator must not know about jaxrens.

Point ``REMOTE_HANDLER_SLURM_TEMPLATES`` at a directory containing both your
cluster header and this body, e.g. by copying it::

    import shutil
    from jaxrens_al import template_dir
    shutil.copy(template_dir() / "body_jaxrens.sh.j2", my_templates_dir)
"""

from pathlib import Path


def template_dir() -> Path:
    """Return the directory holding jaxrens_al's SLURM body templates."""
    return Path(__file__).resolve().parent


__all__ = ["template_dir"]
