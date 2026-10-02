.PHONY: help setup models check smoke extract structured ingest ingest-nodense ask eval eval-fast ingest-all external fmt lint test clean-cache

.DEFAULT_GOAL := help
BACKEND ?= hybrid

help:             ## list targets (variables: BACKEND=native|docling|hybrid, Q="...", DEBUG=1, EMBED=<hf id>, VLM=0|1)
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:            ## one-command setup: uv, Tesseract, LibreOffice, pinned deps, .env, models, `ae check` (re-runnable)
	./setup.sh

models:           ## pre-download the embedding model (EMBED, + bge-small fallback) and Docling models into the HF cache
	uv run python -c "from ae.models import prefetch; prefetch('$(EMBED)' or None)"

check:            ## verify tesseract, libreoffice, packages, uv.lock, .env + API key, cached models, indexes
	uv run ae check

smoke:            ## ~6 min end-to-end check on one file of each type (ingest + ask + score 9 dev questions)
	uv run ae smoke --backend $(BACKEND)
	uv run ae ask "What is the maximum discharge current rating of the EV-BMS-100?" --backend $(BACKEND) --name smoke

extract:          ## run extraction only, dump data/extracted/$(BACKEND)/
	uv run ae extract --backend $(BACKEND)

structured:       ## load CSV/XLSX into SQLite and print the schema
	uv run ae load-structured

EMBED ?= ibm-granite/granite-embedding-english-r2
VLM ?= 1

ingest:           ## extract + structured + numerals + chunk + index + embeddings (+ VLM figure descriptions when VLM=1)
	uv run ae ingest --backend $(BACKEND) $(if $(EMBED),--embed $(EMBED),) $(if $(filter 1,$(VLM)),--vlm,)

ingest-nodense:   ## same, without embeddings (no model download needed)
	uv run ae ingest --backend $(BACKEND)

ask:              ## make ask Q="..."  (add DEBUG=1 for routing/retrieval details)
	uv run ae ask "$(Q)" --backend $(BACKEND) $(if $(filter 1,$(DEBUG)),--debug,)

eval:             ## full staged evaluation for all backends -> data/eval/RESULTS.md (builds any missing index first; LLM calls are cached)
	uv run ae eval

eval-fast:        ## extraction + retrieval stages only (no LLM calls)
	uv run ae eval --no-answers

ingest-all:       ## build the three indexes used by `make eval`
	$(MAKE) ingest BACKEND=native
	$(MAKE) ingest BACKEND=docling
	$(MAKE) ingest BACKEND=hybrid

external:         ## fetch the external evaluation corpus, ingest it (all backends) and score it -> data/eval/EXTERNAL.md
	uv run ae fetch-external
	for be in native docling hybrid; do uv run ae ingest --backend $$be --corpus data/external/evalset --name ext $(if $(EMBED),--embed $(EMBED),) $(if $(filter 1,$(VLM)),--vlm,); done
	uv run ae eval-external --questions gold/external_questions.json

fmt:              ## format and auto-fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint:             ## format check, lint, type check
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy ae

test:             ## unit tests (routing tests need `make ingest BACKEND=native`)
	uv run pytest -q

clean-cache:      ## delete page caches and extraction dumps (indexes are kept)
	rm -rf data/cache data/extracted
