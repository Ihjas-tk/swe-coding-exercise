.PHONY: setup extract ingest ask eval clean-cache

BACKEND ?= thin

setup:            ## create venv and install pinned deps (needs: uv, tesseract)
	uv sync

extract:          ## run extraction only, dump data/extracted/$(BACKEND)/
	uv run ae extract --backend $(BACKEND)

structured:       ## load CSV/XLSX into SQLite and print the schema
	uv run ae load-structured

EMBED ?= ibm-granite/granite-embedding-english-r2

ingest:           ## extract + structured + numerals + chunk + index (+ embeddings when EMBED is set)
	uv run ae ingest --backend $(BACKEND) $(if $(EMBED),--embed $(EMBED),)

ingest-nodense:   ## same, without embeddings (no model download needed)
	uv run ae ingest --backend $(BACKEND)

ask:              ## make ask Q="..."
	uv run ae ask "$(Q)"

eval:             ## (phase 3) run the full staged evaluation
	uv run ae eval --backend $(BACKEND)

clean-cache:
	rm -rf data/cache data/extracted
