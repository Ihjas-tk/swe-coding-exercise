.PHONY: help setup models check smoke extract structured ingest ingest-nodense ask eval eval-quick eval-fast external fmt lint test clean-cache

.DEFAULT_GOAL := help

help:             ## list targets (variables: Q="...", DEBUG=1, CORPUS=<dir>, NAME=<index>, QUESTIONS=<file>, EMBED=<hf id>, VLM=0|1)
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:            ## one-command setup: uv, Tesseract, LibreOffice, pinned deps, .env, models, `ae check` (re-runnable)
	./setup.sh

models:           ## pre-download the embedding model (EMBED) and the Docling layout models into the HF cache
	uv run python -c "from ae.models import prefetch; prefetch('$(EMBED)' or None)"

check:            ## verify tesseract, libreoffice, packages, uv.lock, .env + API key, cached models, index
	uv run ae check

smoke:            ## optional ~4 min self-test: one file of each type in its own index (ingest + ask + score 16 dev questions)
	uv run ae smoke
	uv run ae ask "What is the maximum discharge current rating of the EV-BMS-100?" --name smoke

extract:          ## run extraction only, dump data/extracted/ (JSON + Markdown per document)
	uv run ae extract

structured:       ## load CSV/XLSX into SQLite and print the schema
	uv run ae load-structured

EMBED ?= ibm-granite/granite-embedding-english-r2
VLM ?= 1

CORPUS_OPTS = $(if $(CORPUS),--corpus $(CORPUS),) $(if $(NAME),--name $(NAME),)

ingest:           ## index patents/ design_docs/ structured/ (or CORPUS=<dir> into NAME=<index>): extract + structured + numerals + chunk + embeddings (+ VLM figure descriptions when VLM=1)
	uv run ae ingest $(CORPUS_OPTS) $(if $(EMBED),--embed $(EMBED),) $(if $(filter 1,$(VLM)),--vlm,)

ingest-nodense:   ## same, without embeddings (no embedding model needed)
	uv run ae ingest $(CORPUS_OPTS)

ask:              ## make ask Q="..."  (DEBUG=1 for routing/retrieval details; NAME=<index> for a custom corpus)
	uv run ae ask "$(Q)" $(if $(NAME),--name $(NAME),) $(if $(filter 1,$(DEBUG)),--debug,)

eval:             ## full staged evaluation of the built-in corpus -> data/eval/RESULTS.md (builds the index first if missing; LLM calls are cached)
	uv run ae eval

eval-quick:       ## answer a question file and score it: make eval-quick QUESTIONS=<file> NAME=<index> (default: the dev set on the default index)
	uv run ae eval --quick $(if $(NAME),--name $(NAME),) $(if $(QUESTIONS),--questions $(QUESTIONS),)

eval-fast:        ## extraction + retrieval stages only (no LLM calls)
	uv run ae eval --no-answers

external:         ## fetch the external evaluation corpus, ingest it (index `ext`) and score it -> data/eval/EXTERNAL.md
	uv run ae fetch-external
	uv run ae ingest --corpus data/external/evalset --name ext $(if $(EMBED),--embed $(EMBED),) $(if $(filter 1,$(VLM)),--vlm,)
	uv run ae eval-external --questions gold/external_questions.json

fmt:              ## format and auto-fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint:             ## format check, lint, type check
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy ae

test:             ## unit tests (integration tests need `make ingest` first and skip without it)
	uv run pytest -q

clean-cache:      ## delete page caches (native, layout, DOCX renders) and extraction dumps (indexes and LLM/VLM/embedding caches are kept)
	rm -rf data/cache/native data/cache/layout data/cache/render data/extracted
