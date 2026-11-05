# syntax=docker/dockerfile:1.7
#
# Verification image for jaxrens_active_learning. Pip-installs the package
# with its git deps and imports the main entry points -- the same
# "install IS the check" probe CI runs (.github/workflows/ci.yml). Useful for
# reproducing the CI install path locally before pushing, or for handing a
# self-contained environment to someone without the sibling checkouts.
#
# The git deps (alviz, remote_handler, jaxrens) are public HTTPS URLs in
# pyproject.toml, so the build needs no SSH key or token:
#
#     docker build -t jaxrens_al:verify .
#
# Override the extras installed (default mirrors CI):
#
#     docker build \
#         --build-arg EXTRAS=dev,run,refdb \
#         -t jaxrens_al:verify .

FROM python:3.13

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/jaxrens_al
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY test/ ./test/

ARG EXTRAS=dev,run

RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Verification probes (mirrors CI's install-OK check + a CLI smoke test).
# The run-side block is conditional: with [run] left out, remote_handler +
# jaxrens are both absent and the run-side imports are skipped rather than
# failing the build. When [run] IS in the extras, we additionally check
# `import remote_handler` and `import jaxrens` directly -- jaxrens_al.sampler
# only imports remote_handler transitively and never imports jaxrens at all
# (jaxrens runs on the cluster), so without explicit probes those packages
# could be broken or missing and the build would still pass.
RUN python <<'PY'
import importlib.util
import jaxrens_al
print(f"jaxrens_al {jaxrens_al.__version__} OK")

run_side = all(
    importlib.util.find_spec(n) is not None
    for n in ("remote_handler", "jaxrens")
)
if not run_side:
    print("run-side ([run] extra) not installed; skipping its checks")
else:
    import remote_handler
    print(f"remote_handler {getattr(remote_handler, '__version__', '?')} OK")
    import jaxrens
    print(f"jaxrens {getattr(jaxrens, '__version__', '?')} OK")
    from jaxrens_al.sampler import JaxrensNSSampler  # noqa: F401
    print("JaxrensNSSampler import OK")
PY

RUN jaxrens-al-dashboard --help > /dev/null && echo "jaxrens-al-dashboard CLI OK"

CMD ["python", "-c", "import jaxrens_al; print(jaxrens_al.__version__)"]
