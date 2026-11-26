# Documentation guidelines

Documentation should make the project easier to understand, use, maintain, and experiment with.

## General principles

- Prioritize clarity, conciseness, and relevance.
- Document what is useful to know, not everything that was done.
- Prefer explanations of non-obvious behavior, assumptions, constraints, and consequential design decisions.
- Avoid duplicating information available in source code, configuration, command-line help, or other documentation.
- Keep documentation proportionate to the complexity and importance of the subject.

## Organization

- READMEs should primarily explain purpose, setup, and usage.
- Dedicated technical documents should explain significant algorithms, architecture, methodology, and design decisions.
- Keep detailed experimental results in structured result files and summarize only their important findings in prose.
- Give each document a clear responsibility.
- Introduce additional documents only when they serve a distinct and useful purpose.

## Maintaining documentation

- Update existing descriptions rather than continually appending new sections.
- Describe the current state of the project by default.
- Remove superseded instructions and obsolete implementation details.
- Consolidate related information and eliminate redundancy.
- Reassess the relevance of existing content when making substantial documentation changes.
- Do not maintain development transcripts in documentation.

## Experiments and research

- Document consequential numerical assumptions, experimental methodology, results, limitations, and conclusions.
- Preserve meaningful historical results and algorithm revisions when they support reproducibility or future
  investigation.
- Distinguish measured observations from hypotheses and expectations.
- Document unsuccessful experiments when their findings are useful for avoiding repeated work or informing future
  decisions.
- Keep historical results distinct from current performance claims.
- Prefer structured, reproducible artifacts over extensive descriptions of individual experimental runs.

## What not to document

Unless specifically useful for future development, omit:

- Chronological implementation and migration histories.
- Routine refactorings and incremental implementation changes.
- Temporary debugging observations.
- Routine test counts and successful tooling checks.
- Exhaustive inventories of classes, functions, and parameters.
- Information already adequately preserved in Git history.
- Speculative functionality and unimplemented plans.

## Writing style

- Use direct, precise, technically accurate language.
- Prefer short sections with descriptive headings.
- Use tables and examples where they improve comprehension.
- Avoid unnecessary repetition, filler, and excessive qualification.
- Ensure documented commands and references remain valid.
- Wrap Markdown prose at 120 characters, consistent with the project's code formatting convention.
- Format new and modified Markdown documents using mdformat.
- Allow longer lines where wrapping would compromise readability or Markdown syntax.
- Avoid unnecessary line breaks and blank lines that do not serve a clear structural or readability purpose.

Documentation is a maintained reference, not an append-only record of development activity.
