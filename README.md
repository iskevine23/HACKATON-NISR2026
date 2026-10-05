# Ntawusigara

**District targeting for poverty reduction and financial inclusion in Rwanda**

*Ntawusigara* means "no one is left behind". The tool shows where poverty and financial exclusion overlap across Rwanda's 30 districts, so social protection and financial inclusion programmes can be aimed at the same places.

NISR 2026 Big Data Hackathon, Track 2: Financial Inclusion and Poverty Reduction.

- **Live app:** [add your deployed link]
- **Team:** [Name 1], [Name 2], [University]

## The problem

Rwanda cut poverty from 39.8% in 2017 to 27.4% in 2024 (EICV7), and 92% of adults are now formally served by financial services (FinScope 2024). National averages hide two gaps that matter for NST2:

1. Poverty is concentrated: district rates range from 6.8% (Nyarugenge) to 51.4% (Nyamagabe), and in some districts poverty barely fell after 2017.
2. Financial inclusion is wide but shallow: in several poor districts fewer than 1 in 10 adults has a bank account, and up to 18% rely only on informal finance or none.

Planners in MINECOFIN, MINALOC and LODA, the National Bank, SACCOs and NGOs need to know where these gaps meet. The data exists in three separate NISR publications; Ntawusigara puts it in one place and turns it into a ranking that users can adjust and question.

## What it does

- **Priority index** combining five indicators: poverty rate, number of poor people, adults not formally served, adults without a bank account, and slow poverty reduction since 2017.
- **Adjustable weights:** users set how much each indicator counts and the map and ranking update instantly.
- **Robustness test:** 10,000 random weight sets show which districts are priorities whatever the weights. Robust top 5: Gisagara, Nyanza, Nyamasheke, Nyaruguru, Ngoma.
- **District typologies** from k-means clustering: Poor and under-served (12), Lower poverty, finance gaps (11), Stalled progress (4), Banked urban core (3).
- **District profiles** with poverty trends, a financial access strand and data-driven options to consider.
- **CSV export** of any ranking, with the weights used.
- **Three languages:** the whole interface switches between Kinyarwanda, English and French.
- **On-device AI assistant ("Ask the data"):** ask questions in Kinyarwanda, English or French. A small model trained by the team runs in the browser: no API, no key, no server, and it works offline. Answers are computed from the tool's district table, so figures cannot be invented.
- **Briefing notes:** one click writes a three-paragraph note on any district, in the page language, for district officials and SACCO managers. Generated on the device from the same table.
- **Rank uncertainty:** every rank comes with a likely range. The tool re-draws each district's survey estimates 1,000 times within their sampling error and re-ranks them with the user's weights. Gisagara is first, but its likely rank is 1 to 9; the top six all span roughly 1 to 12. The honest message for planners: the leading districts are a priority *group*, not a strict order.
- **Plan coverage:** enter how many people a programme can reach (and optionally the cost per person). The tool shares them across the top priority districts, or all 30, in proportion to their poor people, capped at each district's poor population, with a CSV download.
- **One-page district report:** prints an A4 brief for any district, in the page language, for district planning meetings.
- **What changed:** ready for EICV8 or the next FinScope. Load the new district figures as a CSV (a template is provided); the tool flags which changes are larger than sampling error and shows how the ranking moves. A clearly labelled simulated demo shows why this matters: sampling noise alone can reshuffle the top 5 while no real change is flagged.
- **Household poverty model (phase 2):** a tested pipeline that trains on EICV7 microdata as soon as access is granted.

## Key findings

- **Ranks are uncertain, and that is a finding in itself.** Allowing for survey sampling error, the six leading districts all have likely ranks spanning about 1 to 12. Gisagara is the clearest priority (top 5 in 81% of simulations); doubling the assumed sample size narrows the ranges but leaves the overlap.

- An estimated 3,629,696 people live below the poverty line (district poverty rates applied to 2022 Census populations).
- Poorer districts have fewer banked adults (Spearman rho = -0.38, p = 0.038) and somewhat more adults outside formal finance (rho = 0.34, p = 0.069).
- Four districts (Nyanza, Ngoma, Burera, Rwamagana) form a "stalled progress" group: poverty fell by under 6 points since 2017, against 12.4 nationally.
- Large-population districts such as Nyagatare, Rusizi and Rubavu each have over 200,000 poor residents: they rise quickly when the number of poor people is weighted more.

## Repository

```
data/raw/            Source tables (EICV7, FinScope 2024, Census 2022) and district shapes
data/processed/      district_index.csv and summary.json (generated)
analysis/            build_index.py: cleaning, index, robustness, clustering
                     district_model_check.py: tests district-level prediction (negative result)
                     train_poverty_model.py: household model for EICV7 microdata (phase 2)
                     train_assistant.py: trains the on-device assistant, writes app/ai-model.js
app/                 Static web app (index.html, data.js, ai-model.js, ai-local.js), deployable as is
docs/                Methodology, data sources, AI disclosure, roadmap
```

## Run it

```bash
pip install -r requirements.txt
python analysis/build_index.py      # rebuilds data/processed/ and app/data.js
python -m http.server -d app 8000   # open http://localhost:8000
python analysis/district_model_check.py
python analysis/train_assistant.py      # retrains the assistant (standard library only)
python analysis/rank_uncertainty.py     # rank intervals with default weights (--n-eff to test the assumption)
python analysis/train_poverty_model.py --demo   # synthetic smoke test only
```

## Deploy

The whole app, including the AI assistant, is static: no API key, no server, no running costs. Any static host works.

- **GitHub Pages:** push the repo, then Settings > Pages > Deploy from branch, folder `/app` (or copy `app/` to `docs/` and choose `/docs`).
- **Netlify or Vercel:** set the publish directory to `app`, with no build command (`vercel.json` already does this).
- **Offline:** open `app/index.html` directly from a USB stick or laptop; everything works without internet except the web fonts.

## The on-device assistant

`analysis/train_assistant.py` generates about 2,400 example questions in the three languages, trains two multinomial Naive Bayes classifiers on character n-grams (one for the type of question, one for its language) and writes the weights to `app/ai-model.js` (about 125 KB). `app/ai-local.js` then finds districts, provinces, indicators, thresholds and direction in the question with rules, and computes the answer from the district table, using the user's current weights for anything about priority. Thirteen question types are covered: rankings, district profiles, comparisons, provinces, typologies, priorities, threshold filters, national figures, correlation, method, briefing notes, greetings, and polite refusal of off-topic questions. To teach it new phrasings, add examples in `train_assistant.py` and rerun it.

## Methodology and limits

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md). In short: district estimates come from samples of roughly 400 to 500 households or adults, the 2017 poverty rates are NISR model estimates, and FinScope district values were read from the published chart.

## Data

NISR data is used under the Creative Commons Attribution 4.0 licence. Full citations in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## AI use

See [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md), as required by the hackathon rules.
