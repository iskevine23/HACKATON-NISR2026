# Methodology

## 1. Data

| Source | Level | Indicators used |
|---|---|---|
| NISR, EICV7 Poverty Profile 2023/24, Annex B | 30 districts | Poverty headcount 2024; modelled 2017 rate |
| AFR and NISR, FinScope Rwanda 2024, Figure 13 | 30 districts | Access strand: banked, other formal only, informal only, excluded (adults 16+) |
| NISR, Census 2022 (RPHC5) | 30 districts | Resident population, urban share |

The three tables are joined on district name with a one-to-one check (`validate="1:1"`). Census populations sum to the national total of 13,246,394.

## 2. Derived indicators

| Indicator | Definition | Direction |
|---|---|---|
| Poverty rate | EICV7 headcount, 2024 | Higher = higher priority |
| Number of poor people | Poverty rate x 2022 population | Higher = higher priority |
| Not formally served | Informal only + excluded (% of adults) | Higher = higher priority |
| Unbanked | 100 - banked (% of adults) | Higher = higher priority |
| Slow progress | Largest district fall since 2017 minus this district's fall | Higher = higher priority |

Each indicator is min-max scaled across the 30 districts to the range 0 to 1.

## 3. Priority index

`index = 100 x sum(w_i x scaled_i) / sum(w_i)`

Default weights: poverty rate 0.35, number of poor 0.20, not formally served 0.20, unbanked 0.10, slow progress 0.15. Poverty carries the most weight because Track 2 targets poverty reduction; the two finance indicators together carry 0.30. Users can change every weight in the app.

## 4. Robustness

Weights are a value judgement, so we draw 10,000 weight vectors from a flat Dirichlet distribution (all weightings equally likely) and record how often each district ranks in the top five. A district above 50% is a priority under most reasonable weightings. Random seed: 2026.

## 5. Typologies

K-means (k = 4, 50 initialisations, standardised features) on poverty rate, share not formally served, share banked and poverty fall since 2017. Clusters are named by rules applied to their centres:

- **Stalled progress:** mean poverty fall under 7 points
- **Poor and under-served:** poverty at least 30% and not formally served at least 9%
- **Banked urban core:** banked at least 40%
- **Lower poverty, finance gaps:** the remaining districts with lower poverty

We chose k = 4 for interpretability with 30 districts; the typology is descriptive, not a statistical finding.

## 6. Options to consider

District profiles show up to three prompts triggered by simple rules (for example, more than 12% of adults not formally served). The suggested responses come from FinScope 2024's own recommendations and barriers (phone ownership as the main barrier to mobile money; collateral as a barrier to credit). They are discussion prompts, not evaluated interventions.

## 7. Machine learning

### District-level prediction: tested and not used

`analysis/district_model_check.py` predicts district poverty from the five finance and urban indicators with leave-one-out cross-validation:

| Model | LOO R² | LOO MAE (points) |
|---|---|---|
| Predict the mean (baseline) | 0 | 11.1 |
| Ridge regression | -0.048 | 10.91 |
| Random forest | -0.019 | 10.36 |
| Gradient boosting | 0.247 | 7.8 |

Only gradient boosting beats the baseline, and with 30 observations and several model and feature choices tried, that result is fragile. We therefore do not rank districts with predictions. Prediction belongs at household level.

### Household poverty model (phase 2)

`analysis/train_poverty_model.py` trains a survey-weighted logistic regression (interpretable, exported to the app for what-if scenarios) and a gradient boosting benchmark on EICV7 households, using variables NISR itself uses to model poverty. It reports 5-fold cross-validated AUC, Brier score and accuracy, and compares predicted and official district rates. It has been tested end to end on synthetic data only (`--demo`); no results will be reported until it runs on the real microdata.

### Rank uncertainty

District estimates come from samples of about 400 to 500 households (EICV7) or adults (FinScope 2024). Neither source file in `data/raw/` carries standard errors, so the tool assumes an effective sample size of 250 per district, roughly halving the nominal sample for the design effect of cluster sampling. In each of 1,000 draws (2,000 in `analysis/rank_uncertainty.py`), every district's 2024 and modelled 2017 poverty rates are drawn from a Beta distribution and its four FinScope shares from a Dirichlet distribution, both centred on the published estimate with that sample size. Derived indicators, the 0-1 rescaling and the weighted index are recomputed in each draw, and the reported "likely rank" is the 5th to 95th percentile of the district's rank across draws. Population is treated as fixed (census).

Result with default weights: Gisagara 1 to 9 (top 5 in 81% of draws); Rusizi, Nyanza, Nyamasheke, Nyamagabe and Nyaruguru each 1 to 11 or 12. With an effective sample of 500 the ranges narrow (Gisagara 1 to 7) but the leading group still overlaps. The assumption is the main limitation: add NISR's published standard errors as `poverty_2024_se` (percentage points) and FinScope's district sample sizes as `finscope_n` to the raw files and they are used automatically.

### Comparing survey rounds

A change between rounds is flagged when it exceeds 1.96 standard errors of the difference, using the same effective sample size for both rounds. The new ranking is recomputed with the same index; "slow poverty reduction" is still measured against the 2017 modelled baseline. This is a screening rule for 30 districts at once, so a few flags can be expected by chance.

### Coverage allocation

People are shared in proportion to each district's number of poor people (top N districts by current priority rank), or to poor people times priority score (all districts), and never above a district's poor population; any excess is redistributed, then unplaceable people are reported. It is a starting point for discussion: final targeting should use sector and household information.

### AI assistant

The assistant runs entirely in the browser and uses no external AI service. A multinomial Naive Bayes classifier on character 3- and 4-grams, trained by `analysis/train_assistant.py` on about 2,400 generated questions in Kinyarwanda, English and French, recognises the type of question (13 classes); a second classifier detects the language. Rules then extract districts, provinces, indicators, thresholds and ranking direction, and the answer is computed from the tool's own district table, including the user's current weights. Briefing notes are written the same way from templates in three languages. Because every figure is read from the table, the assistant cannot invent numbers or causes; its failure mode is misreading a question, which the interface discloses. On a set of 23 hand-written questions not used in training, the classifier labels 21 correctly, and the extraction rules correct the remaining two.

## 8. Limitations

- **Sampling error.** EICV7 interviews about 500 households per district and FinScope at least 400 adults per district. Differences of a few points between districts may not be statistically significant.
- **Modelled baseline.** EICV7 changed the poverty methodology, so the 2017 rates are NISR model estimates; NISR is 95% confident of a fall in 17 of 30 districts only.
- **Different populations.** Poverty rates cover all individuals; FinScope covers adults 16+.
- **Chart-read values.** FinScope district values were read from Figure 13 of the report. Nyarugenge's row shows three values; we assigned 0% informal only and 1% excluded so the row sums to 100. Checking against FinScope microdata is the first step in our roadmap.
- **Population year.** Poor people counts apply 2024 rates to 2022 populations, so they slightly understate current numbers.
- **Correlation is not causation.** The tool shows where gaps overlap, not why.
