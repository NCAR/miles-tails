# Review Paper

Run a structured multi-pass review of the LaTeX paper in the current project.
The argument should be the path to the .tex file, e.g.: `/review-paper FFS_for_hurricanes/main_james2.tex`
If no argument is given, look for the most recently modified .tex file in the project.

## Pass 1 — Adversarial read

Read the abstract and conclusions. For each claim:
- Is it supported by results shown in the paper?
- Is the framing defensible (no overclaims, hedged where needed)?
- Are there any sentences a reviewer could attack as unsupported?

Report specific line numbers and the concern.

## Pass 2 — Consistency sweep

Cross-check the following against the actual figures and tables in the paper:
- All numbers cited in the body text (rates, probabilities, ratios, speedups)
- All figure cross-references (`\ref{fig:...}`) — do the labels exist and are they cited in the right context?
- Table values match what the text says about them
- Storm-specific claims: dates, coordinates, genesis designations

Report any mismatch with the line number of the claim and the line number of the ground truth.

## Pass 3 — Unresolved placeholders

Search the .tex and .bib files for:
- `XXXX`, `TODO`, `TBD`, `\textbf{[`, `update.*before submission`
- Grant numbers that are still placeholder (e.g. `AGS-XXXX`)
- URLs marked as needing update
- `documentclass` or `bibliographystyle` still set to draft values (`article`, `plainnat`)

List each one with file and line number. Note which require user input vs. can be resolved automatically.

## Pass 4 — Bibliography audit

- Every `\cite{}` key in the .tex must exist in the .bib
- Every entry in the .bib must be cited in the .tex (flag dead entries)
- Entries missing year, journal/venue, or DOI/URL
- Keys where the year in the key name does not match the `year = {}` field
- arXiv entries: confirm `year` reflects when the paper went public, not just submission date

## Pass 5 — Narrative coherence

Does the paper tell one clean story from abstract → intro → methods → results → discussion?
- Are the case studies introduced consistently and discussed in the same order throughout?
- Does the discussion section address the same claims made in the results?
- Are limitations stated in the discussion consistent with what was actually done?

## Output format

For each pass, report:
1. A one-line verdict (PASS / ISSUES FOUND)
2. A bulleted list of specific findings with file:line references
3. Which findings are blockers for submission vs. minor polish

Do not fix anything — report only. Ask the user which items to address before making edits.
