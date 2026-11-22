SHELL=/bin/bash

diff-to-clipboard:
	@git add -N .
	@git diff | wl-copy

head-to-clipboard:
	@git show HEAD | wl-copy
