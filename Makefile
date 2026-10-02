.PHONY: setup data run figures report notebook test lint all

# One BLAS thread per worker: the pipeline already parallelises across campuses.
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1

setup:  ## install the locked environment
	uv sync --locked

data:  ## download Facebook100 (207 MB, idempotent)
	uv run fb100 data

run:  ## every analysis on the 100 campuses -> results/ (about 20 min with 3 workers)
	uv run fb100 run

figures:  ## static README figures -> docs/figures/
	uv run fb100 figures

report: figures  ## static report -> site/index.html
	uv run fb100 report

notebook:  ## re-execute the walkthrough notebook in place
	JUPYTER_PATH=.venv/share/jupyter uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: data run report
