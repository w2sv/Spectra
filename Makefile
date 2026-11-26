SHELL=/bin/bash

.PHONY: format-markdown

format-markdown:
	@git ls-files -z '*.md' | xargs -0 uv run mdformat
