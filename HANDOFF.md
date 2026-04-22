# Handoff Checklist

Everything you need to do before the repo goes public. Work through top to bottom. When every box is checked, the repo is ready to push to GitHub.

---

## Integrity — do not skip

These items exist to protect your integrity as a researcher and a candidate. Skipping any of them risks either factual errors in the public record or credibility damage if an interviewer probes the work.

- [ ] **Verify all 15 citations manually.** Open `CITATION_CHECKLIST.md` and work through each entry. AI-generated bibliographies hallucinate references at significant rates; every single citation must be confirmed against Google Scholar or the publisher's website before the README's AI-disclosure claim ("citations were verified manually") is accurate. Budget 2–3 hours for this. Do not shortcut.

- [ ] **Review the AI disclosure language in the README.** Currently Option C: *"This project used AI tools throughout, including for code authorship assistance on portions of the machine learning pipeline, literature discovery, and prose drafting. The authors designed the research question, constructed the data pipeline's identification strategy (including the lookahead guard), selected the econometric framework, interpreted results, and bear responsibility for all claims in the paper."* Confirm this is accurate. If AI assistance was broader or narrower, revise.

- [ ] **Confirm co-authors are okay with publication.** The paper has three authors (you, Ken Hayashi, Taiki Takahashi). Before pushing, get their sign-off on (1) the repo being public under your name, (2) the MIT license, and (3) the AI disclosure language. If either co-author is uncomfortable, hold off.

- [x] **Run the pipeline end-to-end with your real data.** ✅ VERIFIED — pipeline reproduces the paper's Table 1 bit-identically. All seven experiments' betas, p-values, OOS R², DM statistics, and directional accuracies match the paper exactly. All six figures generated cleanly. Logged output saved.

- [ ] **Run the test suite with pytest (not IDE runfile).** From a terminal, not Spyder's runfile: `cd fx-gap-repo && pip install pytest && pytest tests/ -v`. Spyder's `runfile` only imports the test module; it does not execute the test functions. If any fail, investigate — they're checking the invariants the paper's identification strategy depends on.

## Pre-push cleanup

- [ ] **Remove `CODE_REVIEW_NOTES.md` if you prefer it private.** It contains review observations that are useful as internal documentation but not necessary for public consumption. Two options: (1) keep it in the repo as a transparency signal, (2) move it to a private gist and keep the repo clean. Either is defensible; most candidates keep internal review docs private. Your call.

- [ ] **Remove `PLAN.md` similarly.** This was project-execution scaffolding; it's not relevant to a public reader. Consider removing.

- [ ] **Remove `src/_original_paper_code.py`.** The original monolith is preserved in the repo for traceability during the refactor. Before publishing, decide: keep it (as a "before" artifact for readers curious about the refactor) or remove it. Removing is cleaner; keeping is more transparent. Default to removing unless you want to tell the refactor story explicitly.

- [ ] **Double-check `.gitignore` before the first commit.** Specifically: run `git status` after `git init && git add .` and confirm that no files under `data/` (other than `data/README.md`) are staged. If you see `USDJPY.csv` or `JPEquities.xlsx` listed, stop — the .gitignore isn't catching them.

- [ ] **Set up the GitHub repo's About section.** A one-liner description plus the paper's topic as tags (`econometrics`, `fx`, `japanese-equities`, `microstructure`, `hac-ols`) makes the repo discoverable and scannable.

## Code polish (optional but recommended)

- [ ] **Run `ruff check src/ tests/`** to catch any style issues. `pip install ruff` first if needed. Fix whatever it flags.

- [ ] **Run `mypy src/`** as an optional type-correctness check. Lots of warnings for pandas DataFrame types are expected and fine.

- [ ] **Consider adding a GitHub Actions workflow** for CI. A minimal `.github/workflows/test.yml` that installs requirements and runs `pytest` on push produces a green badge you can display in the README. Worth the 15 minutes.

## Website integration

- [ ] **Update your portfolio site's project card** for this project to link to the GitHub repo. The `docs/website_summary.md` file is drafted as suitable portfolio copy — adapt to your site's markdown/HTML conventions.

- [ ] **Verify the GitHub link on your portfolio site loads cleanly** once the repo is public. The 404 between publish and link update is a common small embarrassment.

- [ ] **If you're putting the PDF online**, host the version currently in `paper/` (not a reprint from the original submission). The repo and the paper should be the same version.

## Known items to mention in interviews

These are things an interviewer may ask about, based on a careful read of the repo and paper. Have answers ready.

- **The 60.1% directional accuracy on Mitsubishi.** Be ready to explain that this is 1 of 7 configurations, the others cluster near 50%, and you interpret it as likely overfitting / favorable sample variance, not a robust signal. The paper's own Section 5.4 flags this.

- **Negative out-of-sample R² across all experiments.** Be ready to explain that the contemporaneous HAC betas are the paper's headline result, that OOS magnitude prediction is expected to fail for a bounded information-friction effect, and that the negative R² is itself evidence the effect is behaving as predicted (not a counterexample).

- **The Granger null.** Be ready to explain that Granger causality failing at lags 1+ is exactly what the "Trading Place Hypothesis" predicts — the information is absorbed at the opening auction, not trickling through subsequent sessions.

- **The 10^-5 gap truncation rule.** Know roughly how many observations it affects and why the threshold was chosen. Ideally have a robustness check in hand (re-run with 10^-4 and 10^-6 and confirm similar results).

- **AI usage.** Know what you actually did versus what AI did. If asked "did you write the lookahead guard yourself," you should be able to say yes or no precisely, not defensively.

- **The quanto extension in Section 7.** Know that it's described as a research direction, not a worked result. If asked for concrete quanto pricing, be honest that the pricing model isn't implemented in the repo.

## Ship it

When every relevant box above is checked, the repo is in good shape. Push to GitHub, link from your portfolio site, and add the repo URL to your resume if you haven't already.

Good work. This is the kind of portfolio artifact that differentiates candidates at the shops you're targeting.
