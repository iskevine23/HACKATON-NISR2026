# Data sources

1. National Institute of Statistics of Rwanda (2025). *Seventh Integrated Household Living Conditions Survey (EICV7): Poverty Profile Thematic Report 2023/24.* Annex B: Headcount poverty rate in 2024 (actual) and 2017 (modelled) by district. https://www.statistics.gov.rw/sites/default/files/documents/2025-07/EICV7_Poverty%20Profile.pdf

2. Access to Finance Rwanda, with MINECOFIN, the National Bank of Rwanda and NISR (2024). *Rwanda FinScope Survey 2024.* Figure 13: Financial access strand by district. https://statistics.gov.rw/sites/default/files/documents/2024-09/Rwanda-Finscope-2024-Report_compressed.pdf

3. National Institute of Statistics of Rwanda (2023). *Fifth Rwanda Population and Housing Census 2022.* Resident population by residence, province and district. https://statistics.gov.rw/statistical-publications/population-size-and-population-characteristics

4. District boundaries: Rwanda district boundaries GeoJSON (DLUP source), simplified and projected to SVG paths for the app.

NISR data and analysis are licensed under Creative Commons Attribution 4.0 International.

## Verification checklist

- [ ] Poverty rates match EICV7 Annex B for all 30 districts
- [ ] FinScope access strand values checked against the report figure (and microdata if access is granted)
- [ ] Census populations sum to 13,246,394
- [ ] District boundary source confirmed and credited


## Optional columns

- `poverty_2024_se` in `eicv7_district_poverty.csv`: standard error of the 2024 district poverty rate, in percentage points, if available from NISR.
- `finscope_n` in `finscope2024_district_access.csv`: FinScope 2024 sample size (adults) per district.

When present, these replace the default effective sample size (250) in the rank-uncertainty and round-comparison calculations.
