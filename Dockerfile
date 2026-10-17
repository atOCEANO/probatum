# Cross-version gate: build the one abi3 wheel, then prove it imports and passes
# the whole suite on Python 3.13 and 3.14. The wheel is built once in `builder`;
# each test stage installs that same wheel on a different interpreter.
#
# Run one stage at a time, any failure fails the build:
#   docker build --target test313 -t probatum-gate313 .
#   docker build --target test314 -t probatum-gate314 .
#
# The rust-checks stage is the other half of the same gate: it runs the pure-Rust
# format, lint, and test commands CI runs, so a Rust change is verified the same
# way a Python one is, with no toolchain on the host:
#   docker build --target rust-checks .
#
# The bench stage is opt-in and prints throughput; it is not a correctness gate,
# so its timing never fails a test build:
#   docker build --target bench   .

FROM python:3.13-slim AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt maturin
WORKDIR /src
COPY . .
RUN maturin build --release --locked --out /wheels

# The pure-Rust half of CI. rust-toolchain.toml pins only the channel, so the two
# components are added here; declaring them in that file makes rustup install them
# into an already-provisioned toolchain, which conflicts on some CI runners. The
# base carries Python because clippy over probatum-py needs an interpreter for pyo3's
# build config.
FROM python:3.13-slim AS rust-checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add rustfmt clippy
WORKDIR /src
COPY . .
RUN cargo fmt --all --check \
    && cargo clippy --locked --workspace --all-targets -- -D warnings \
    && cargo test --locked -p probatum-core -p probatum-engine

FROM python:3.13-slim AS test313
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
# the Dockerfile, so the pages can be checked against the stages they name, and
# the tools, so a page has to explain each one. /devtools rather than
# /dev, which is the device filesystem on any linux box
COPY Dockerfile /Dockerfile
COPY dev /devtools
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt /wheels/*.whl \
    numpy gymnasium pandas pyarrow polars optuna cloudpickle pytest \
    && python -c "import probatum; print('py313 wheel import OK')" \
    && pytest -q /tests

FROM python:3.14-slim AS test314
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
COPY Dockerfile /Dockerfile
COPY dev /devtools
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt /wheels/*.whl \
    numpy gymnasium pandas pyarrow polars optuna cloudpickle pytest \
    && python -c "import probatum; print('py314 wheel import OK')" \
    && pytest -q /tests

FROM python:3.13-slim AS bench
COPY --from=builder /wheels /wheels
COPY benchmarks /benchmarks
RUN pip install --no-cache-dir /wheels/*.whl numpy \
    && python /benchmarks/throughput.py

# Per-surface throughput (Engine, an array and a Strategy through backtest, Backtester,
# tune, VectorEnv, Batch).
# Needs the full deps, so it is its own stage; run it and read stdout:
#   docker build --target bench-surfaces -t probatum-bench-surfaces .
#   docker run --rm probatum-bench-surfaces
FROM python:3.13-slim AS bench-surfaces
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl numpy gymnasium optuna cloudpickle
COPY benchmarks /benchmarks
CMD ["python", "/benchmarks/surfaces.py"]

# The tests marked slow, which run at length. Chief among them is the engine
# against a second implementation of the same rules, over randomised paths.
# `tests/reference/simulator.py` is written from the Specification and never from
# the Rust, which is the whole point of it, so a rule the engine gets wrong is a
# rule the reference gets right and the two part company on a number. A case is
# compared only where the reference applied no rule the engine has yet to take up,
# which the reference's own count of the rules it applied decides.
#
# Six seeds of five hundred cases has to be exact: the tolerance is relative 1e-6,
# any disagreement fails the build, and so does a rule the two share that no
# compared case reached. Every gate stage above runs the same comparison over a
# few hundred cases.
#
# `batch_differential.py` is the same idea one tier up, over `Batch`: every
# reinforcement learning number comes out of `reset_at_each`, `advance` and
# `snapshot`, and what stood behind them was a Rust test comparing the batch
# against the engine it is built from, which shares every mistake the engine
# makes. Each env is compared against a reference started at that env's own
# offset, because an episode beginning part way into the series still funds on
# the bars the SERIES funds on (ADR 0020) and nothing that starts
# every env at bar zero can see that rule at all.
#   docker build --target test-slow .
FROM python:3.13-slim AS test-slow
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
COPY Dockerfile /Dockerfile
COPY dev /devtools
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt /wheels/*.whl \
    numpy gymnasium pandas pyarrow polars optuna cloudpickle pytest \
    && pytest -q --slow -m slow /tests

# The chart layer's other half. Every assertion in the correctness gate reads the
# JSON spec, because the gate has no browser and a spec assertion is the sharper
# test of what Python decided (ADR 0042). What it cannot reach is whether the
# shipped JavaScript parses, runs and draws. This stage supplies the browser; like
# test-sb3 it stays out of the local gate because its image is heavy, and CI runs
# both:
#   docker build --target test-browser .
# The image ships the browsers under /ms-playwright but not the python package,
# and the two are versioned together, so the pin here has to match the tag above.
# Its own Python is older than the wheel's floor, so uv puts a 3.13 beside it and
# the browsers stay the image's. That playwright pins a greenlet with no wheel for
# 3.13, which does not build there either, so the constraints file's greenlet
# overrides the pin.
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy AS test-browser
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt uv \
    && grep '^greenlet==' /constraints.txt > /override.txt \
    && uv venv --python 3.13 /venv \
    && uv pip install --python /venv -c /constraints.txt --override /override.txt \
    /wheels/*.whl numpy pandas pytest playwright==1.47.0 \
    && /venv/bin/pytest -q -p no:cacheprovider /tests/test_render.py

# Every chart image in the documentation, rebuilt from the frozen sample data and
# screenshotted. Opt-in, because it needs the published dataset
# mounted and it writes into the tree rather than asserting anything:
#   docker build --target charts -t probatum-charts .
#   docker run --rm -v "<sample-market-data>/data:/data:ro" \
#       -v "${PWD}/.Documentation:/out" probatum-charts
# It shares test-browser's base, so it pulls no image that stage has not already
# pulled: the browsers ship in it and pip cannot install them on its own. It takes
# its Python 3.13 and greenlet the same way, since a different base would mean
# different fonts and so different images.
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy AS charts
COPY --from=builder /wheels /wheels
COPY dev/charts /charts
# the chart asks for `system-ui, -apple-system, "Segoe UI", sans-serif` and a
# container has none of the first three, so every label in every image is drawn
# by whatever this box calls sans-serif. This image calls it WenQuanYi Zen Hei, a
# CJK face whose latin glyphs are narrower and lighter, so without this line the
# whole documentation set silently re-renders in a font nobody chose. DejaVu wins
# the fallback as soon as it is present, and it is what the images already use
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir -c /constraints.txt uv \
    && grep '^greenlet==' /constraints.txt > /override.txt \
    && uv venv --python 3.13 /venv \
    && uv pip install --python /venv -c /constraints.txt --override /override.txt \
    /wheels/*.whl numpy pandas pyarrow playwright==1.47.0
CMD ["sh", "-c", "/venv/bin/python /charts/build.py && /venv/bin/python /charts/shoot.py"]

# The hand-made diagrams, rendered from their mermaid sources. The
# other half of the same idea as `charts`: every picture in the documentation is
# produced by something committed here, so none of them is a file nobody can
# remake. Opt-in, and it writes into the tree rather than asserting anything:
#   docker build --target diagrams -t probatum-diagrams .
#   docker run --rm --shm-size=1g -v "${PWD}/dev/diagrams:/diagrams" \
#     -v "${PWD}/.Documentation/imgs:/out" probatum-diagrams
# The sources are mounted rather than copied in, so a rerun draws what is on disk.
# Baked in at build time they made the second command alone redraw the previous
# build's sources and print the same success lines doing it.
# The bundled headless-shell in this image is broken with an ENOENT, which is why
# puppeteer.json points executablePath at the chromium the image also ships.
FROM minlag/mermaid-cli:11.17.1@sha256:062edb08dcc7f95841c15620241b6934af93aa75c27f223ebe2e81fd0b4da4c9 AS diagrams
USER root
# the image's entrypoint is mmdc itself, with its own puppeteer config bolted on,
# so it has to be cleared to run a loop; mmdc is not on PATH, hence the full path
ENTRYPOINT []
# no -w: the width is mermaid's own, and setting one made every wide diagram
# 2304 where the committed pair are 2352. `-b transparent` rather than white, or
# the images invert badly against GitHub's dark mode
CMD ["sh", "-c", "set -e; for f in /diagrams/*.mmd; do n=$(basename \"$f\" .mmd); /home/mermaidcli/node_modules/.bin/mmdc -i \"$f\" -o \"/out/$n.png\" -c /diagrams/config.json -p /diagrams/puppeteer.json -b transparent -s 3; echo \"rendered $n\"; done"]

# Stable-Baselines3 integration: install torch and sb3 and run the adapter tests.
# torch is heavy, so this stays out of the local gate, and CI runs it:
#   docker build --target test-sb3 .
# torch alone comes from its CPU index, which does not carry every pinned version
# of torch's own dependencies; those install from PyPI with the rest
FROM python:3.13-slim AS test-sb3
COPY dev/constraints.txt /constraints.txt
RUN pip install --no-cache-dir --no-deps -c /constraints.txt torch \
    --index-url https://download.pytorch.org/whl/cpu
COPY --from=builder /wheels /wheels
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
RUN pip install --no-cache-dir -c /constraints.txt /wheels/*.whl \
    numpy gymnasium stable-baselines3 pytest \
    && pytest -q /tests/test_sb3.py

# The source archive a release ships, installed the way pip installs it where no
# wheel fits, then tested, so a file it leaves out fails here. CI runs it:
#   docker build --target test-sdist .
# PIP_CONSTRAINT rather than -c, because only the environment variable reaches
# the isolated environment pip builds the archive in, where maturin is installed.
FROM builder AS test-sdist
RUN maturin sdist --out /sdist
COPY tests /tests
COPY .Documentation /docs
COPY README.md /README.md
COPY Dockerfile /Dockerfile
COPY dev /devtools
WORKDIR /
RUN PIP_CONSTRAINT=/constraints.txt pip install --no-cache-dir /sdist/*.tar.gz \
    numpy gymnasium pandas pyarrow polars optuna cloudpickle pytest \
    && pytest -q -p no:cacheprovider /tests

# Mutation testing of the two pure-Rust crates: cargo-mutants changes one line at a
# time and reruns their tests, and lists every change the tests did not notice.
# The whole pass is slow and takes a lot of memory, and it has taken the Docker
# daemon down when another build ran beside it, so it is opt-in and run alone:
#   docker build --target mutants -t probatum-mutants .
#   docker run --rm probatum-mutants
# CI runs it on the lines each push changes, where a missed mutant fails the job.
FROM rust-checks AS mutants
RUN cargo install --locked cargo-mutants --version 27.1.0
CMD ["cargo", "mutants", "--package", "probatum-core", "--package", "probatum-engine", "--jobs", "4"]

# Instruction counts for the step loop under Valgrind. A count does not depend on
# the machine the way a timing does, so CI compares it with the commit before and
# fails on a rise of more than 10%. Here it prints the counts:
#   docker build --target bench-instructions .
FROM python:3.13-slim AS bench-instructions
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential valgrind \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install --locked gungraun-runner --version 0.20.0
# a build cannot switch address randomization off, which gungraun does by default
ENV GUNGRAUN_ALLOW_ASLR=yes
WORKDIR /src
COPY . .
RUN cargo bench --locked -p probatum-engine --bench step_instructions

# Miri runs the candle tests in an interpreter that reports undefined behaviour,
# which checks the library's one reinterpretation of memory, the floats behind
# every zero-copy view (ADR 0040). It needs a nightly toolchain, pinned by date and
# named on each command so that rust-toolchain.toml does not win. CI runs it:
#   docker build --target miri .
FROM python:3.13-slim AS miri
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain nightly-2026-10-01 --profile minimal \
    --component miri,rust-src
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo +nightly-2026-10-01 miri setup
WORKDIR /src
COPY . .
RUN cargo +nightly-2026-10-01 miri test --locked -p probatum-core candle::
