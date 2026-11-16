# Data, preprocessing and input contract

## Sources and redistribution

| Source | Used for | Collection code |
|---|---|---|
| [SMARD, Bundesnetzagentur](https://www.smard.de/) | Day-ahead prices, actual load/generation, TSO wind/solar forecasts | `scripts/fetch_data.py` |
| [Energy-Charts, Fraunhofer ISE](https://api.energy-charts.info/) | Day-ahead load forecast | `scripts/fetch_data.py` |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | Older ECMWF/ICON weather features | `scripts/make_pregate_fc.py` |

The study recorded SMARD and Energy-Charts as CC BY 4.0 sources. Verify the applicable source
terms and attribution requirements when collecting or redistributing data; this repository's
Apache license does not replace those terms. Open-Meteo API access/usage terms also apply.
This release distributes user-created forecasts, metrics and code, but not the raw feed archive
or a rehosted weather dataset. No paid data credential is needed by the source collectors.

The source URLs and series IDs are declared in `fetch_data.py`. The historical raw panel was
assembled into `data/hourly_raw.parquet`, indexed by UTC timestamps. Cached raw responses live
under `data/raw/`. Those directories are ignored by Git.

## Numeric model schema

`inputs.json` is an object with these three fields:

| Field | Keys | Shape | Units |
|---|---|---|---|
| `target` | DE-LU hourly day-ahead price | 2040 historical values in research setup | EUR/MWh |
| `past_covariates` | `load_fc`, `wind_fc`, `solar_fc` | Each matches target history length | MW |
| `past_covariates` | `price_be`, `price_fr`, `price_nl`, `price_dk1`, `price_at`, `price_ch`, `price_cz` | Each matches target history length | EUR/MWh |
| `future_covariates` | `load_fc`, `wind_fc`, `solar_fc` | Each has exactly 24 forecast-day values | MW |

Every value must be finite. Negative prices are valid. History is ordered oldest first. The
portable helper accepts 168–2040 hours; the recorded study used 2040 (85 days), so shorter
contexts are an interface convenience rather than the reported protocol. All expected channels
are required. Inputs remain in their original units; do not apply a new per-series standardization
that differs from the library's internal normalization.

At the noon D-1 auction gate, prices through the end of D-1 are known from the prior auction;
the target day-D prices are not. Historical wind/solar features are TSO forecasts. For day D,
pre-auction inference must supply the older-weather-derived estimates under the same keys
`wind_fc` and `solar_fc`. Do not silently insert post-auction TSO forecasts and call the result
a pre-auction prediction. No current day-D neighbouring prices are supplied.

Generate a complete synthetic example with `scripts/make_example_input.py`. Its magnitudes
are illustrative; it contains no market observations. The JSON represents aligned numeric
arrays, not timestamps. The caller must ensure units, delivery dates and feature availability.

## Time conventions

`src/epf/data.py` converts UTC feeds to Europe/Berlin delivery days. Historical DST days are
mapped to 24 bins: duplicate autumn 02:00 hours are averaged and the spring gap is interpolated.
The paper prototype instead skips 23/25-hour days and requires 24 original weather hours.
Hourly prices derived from 15-minute observations are averages, not executable quarter-hour bids.

`src/epf/chronos_io.py` builds model dictionaries. It slices target/neighbour histories strictly
before the forecast day, and substitutes `wind_fc_pre` / `solar_fc_pre` only for the current
future inputs. `with_pregate` adds those columns from `data/pregate_fc.parquet`.

## Weather-to-generation mapping

The 14 geographic points, weather models, variables, date range and first usable forecast date
are explicit in `scripts/make_pregate_fc.py`. The mapper fits LightGBM separately for wind and
solar, refitting monthly on earlier dates. Historical targets are TSO forecast levels. It clips
negative predicted generation to zero. A missing weather feature can remove a delivery day.

Historical Previous Runs offsets do not establish that every upstream response would have
been available at the claimed original collection time. Live paper collection separately checks
timestamps, completeness and the gate deadline and stores timestamped source snapshots.

## What can be rebuilt, and what is missing

Collectors and mapper code are included, so new users can rebuild their own panel and weather
mapping subject to feed access and source terms. Raw snapshots are not included; feeds can
revise history. Weather mapper fitted objects were ephemeral and refit under the declared rules;
they are not packaged as frozen weights. A new rebuild is therefore not a claim of exact original
input reconstruction. Original saved forecasts and reports remain the historical result record.

No Hyperliquid/0xArchive data contributed to this released electricity model. The earlier crypto
collection idea is outside the study's reported scope.
