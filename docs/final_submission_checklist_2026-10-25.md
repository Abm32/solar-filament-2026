# Final submission checklist (host requirements as of 2026-10-25)

Sources: Overview › Final Submission (Aug 7), Organizers' Remarks thread (Oct 23 comment),
threads 747960, 747399, 749524, 749526, 728302. Deadline **Nov 15, 2026, 11:59 pm CDT**
(= Nov 16 10:29 IST). Form can be edited until then; the host reads the **last** uploaded file.

## What the host asks for

| requirement | source | status |
|---|---|---|
| Kaggle: one selected final submission | Submissions page | DONE — `submission_v84.csv` selected (1/1) |
| Google Form submitted (one per team) | Overview | **USER** — link below |
| Form: submission file = exact CSV sent to Kaggle | form field | ready: `submissions/submission_v84.csv` (sha 2925e2a2…) |
| Form: 4-page PDF report in the host Overleaf/ACM template | form field | ready: `report/acm/main.pdf` after city fix |
| Form: public Git URL, no permission needed | form field | **USER** — repo is private and unpushed |
| Repo: `requirements.txt` with versions | Overview | DONE |
| Repo: notebook illustrating the entire pipeline | Overview | DONE `notebooks/reproduce_v84.ipynb` (verified on VM: row hash matches) |
| Repo: weights stored in repo and loaded by the notebook (Oct 23 clarification; otherwise "not considered for award") | Organizers' Remarks | DONE `weights/` + SHA256SUMS, loaded by notebook |
| Repo: no additional file needed to reproduce | Overview | DONE (data = Kaggle competition data only) |
| Repo public immediately at close, stays public until winners announced (Nov 30) | Open-Access Policy | **USER** |
| Code modular, documented; not everything in one notebook | thread 728302 | DONE (scripts/ modules + notebook + tests) |
| Report: every micro-decision documented (splits, resize ratios, normalisation, thresholds), cite libraries | thread 728302 | DONE in main.tex; re-read once before upload |
| Report: no sign of AI-generated text without human oversight | Oct 23 remark | user should read and edit in own voice before upload |
| Efficiency: qualitative filter only (not too slow, no memory leak) | thread 749526 | 180 images in 440 s on a T4, flat memory |
| If two selected submissions differ, both must be reproducible | thread 747960 | n/a — one submission selected |

## Google Form fields (prepared answers)

Form: https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform

| field | answer |
|---|---|
| Email | abhi@anukritiai.com (must be reachable; a copy of responses is emailed here) |
| Full names of all team members | Abhimanyu R B |
| Email addresses of all team members | abhi@anukritiai.com |
| Affiliated institutes (optional) | Anukriti AI  ← confirm, same string as report |
| Countries | India  ← confirm |
| GIT URL | https://github.com/Abm32/solar-filament-2026  (after push + public) |
| Submission file | upload `submissions/submission_v84.csv` |
| Technical report | upload `report/acm/main.pdf` |

## User steps, in order

1. Fix the one TODO in the report: `report/acm/main.tex:45` `\city{\todo{City}}` → your city, then rebuild:
   `cd report/acm && docker run --rm -v $PWD:/work -w /work texlive/texlive:latest sh -c 'pdflatex main; bibtex main; pdflatex main; pdflatex main'`
   (must stay ≤ 4 pages; check `grep -c Overfull main.log`).
2. Commit and push (everything is already staged):
   `git commit -m "Final solution: V84 pipeline, weights, reproduction notebook, report, V67–V100 experiment record"`
   `git push origin main`
3. GitHub → repo Settings → General → Danger zone → Change visibility → **Public**.
   Then open https://github.com/Abm32/solar-filament-2026/blob/main/notebooks/reproduce_v84.ipynb in a private window to confirm no login is needed.
4. Fill the Google Form with the table above; upload the CSV and the PDF; submit. Keep the confirmation email.
5. Do not change the selected Kaggle submission afterwards (the form CSV must stay the exact selected file).

## Verified facts behind the checklist

- Selected submission on Kaggle: `submission_v84.csv`, public 0.39, sha256 2925e2a2…, same bytes as `submissions/submission_v84.csv`.
- Notebook end-to-end run on the VM (2026-10-25): 180 images, 440 s, 1112 rows, canonical row hash c5df205e… = submitted file.
- Repo contains no `data/`, `outputs/`, credentials, or public-MAGFiLO-derived files (`git ls-files` audit 2026-10-25).
