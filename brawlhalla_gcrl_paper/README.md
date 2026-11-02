# Brawlhalla GCRL paper

Canonical source: `main.tex`  
Bibliography: `references.bib`  
Compiled paper: `main.pdf`

## Compile

```bash
latexmk -pdf main.tex
```

## Replace projected results

Search for:

```latex
\projected{...}
```

Every projected value renders as `[P:...]` in the PDF, so synthetic/expected values cannot be confused with measured values.

## Evidence status

- Detector metrics: measured / valid.
- Runtime profiling values: measured from repository performance documentation.
- Historical 3-hour retention matrix: measured but explicitly superseded by the later integrity audit.
- Corrected RL results, transfer, ablations, and final match benchmark: projected placeholders until rerun.

## Regenerate current evidence

From the repository root, using the pinned environment:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m brawl.analysis.paper_evidence
```

The command writes PNG and vector PDF figures, a long-form metrics CSV, generated
LaTeX tables, source/output SHA-256 hashes, and an evidence-readiness report to
`brawlhalla_gcrl_paper/evidence/`.

The generator deliberately reads only the current canonical filenames. Historical
`_3h` runs, pilots, failed runs, and alternate versions are not used. Run with
`--strict-readiness` in a release check when missing or under-sampled evidence
should fail the command.
