.PHONY: setup extract ingest ask eval clean-cache

BACKEND ?= thin

setup:            ## create venv and install pinned deps (needs: uv, tesseract)
	uv sync

extract:          ## run extraction only, dump data/extracted/$(BACKEND)/
	uv run ae extract --backend $(BACKEND)

ingest:           ## (phase 2) extract + chunk + index
	uv run ae ingest --backend $(BACKEND)

ask:              ## make ask Q="..."
	uv run ae ask "$(Q)"

eval:             ## (phase 3) run the full staged evaluation
	uv run ae eval --backend $(BACKEND)

clean-cache:
	rm -rf data/cache data/extracted
