# Fuse Energy — commodity watchlist (OilPrice API)

A curated watchlist of OilPrice commodity codes for **[Fuse Energy](https://www.fuseenergy.com)** (London/UK energy retailer), across **petroleum, gas, coal, and marine fuel** (+ carbon). Filtered from the 335 items in these categories down to what a UK energy supplier actually prices, hedges, or references.

- Live prices captured **2026-07-12** (indicative — re-fetch via `get_commodity_price(code)`).
- See [`OILPRICE_API.md`](OILPRICE_API.md) for the full provider/API reference.

---

## 🔥 Core watchlist — highest relevance

The must-watch feeds for a UK gas/power supply book:

| Code | Name | Live price | Why it matters |
|------|------|-----------|----------------|
| `NATURAL_GAS_GBP` | UK Natural Gas | **117.19p / therm** | The UK domestic gas benchmark — top input for a UK supply book |
| `NATURAL_GAS_TTF_SPOT_EUR` | Dutch TTF Spot | **€49.71 / MWh** | European wholesale gas benchmark; drives UK wholesale pricing |
| `BRENT_CRUDE_USD` | Brent Crude | **$75.22 / bbl** | North Sea crude — the UK/Europe oil benchmark |
| `GASOIL_USD` | ICE Low Sulphur Gasoil (Rotterdam) | **$1,043 / tonne** | Heating-oil / diesel benchmark for NW Europe |
| `UK_CARBON_GBP` | UK Carbon Allowances (UK ETS) | **£55.49 / tCO₂** | Compliance + cost driver for a UK supplier |
| `EU_CARBON_EUR` | EU Carbon Allowances (EU ETS) | **€79.20 / tCO₂** | Cross-border carbon reference |

---

## ⛽ Petroleum / oil

| Code | Name | Unit | Note |
|------|------|------|------|
| `BRENT_CRUDE_USD` | Brent Crude Oil | barrel | **core** — UK/Europe crude benchmark |
| `BRENT_SPOT_USD` | Brent Crude Oil Spot | barrel | spot variant |
| `GASOIL_USD` | ICE Low Sulphur Gasoil (Rotterdam) | tonne | **core** — NW Europe distillate |
| `EUROBOB_GASOLINE_USD` | Eurobob Gasoline Futures (ARA) | MT | ARA gasoline (no live price returned) |
| `URALS_CRUDE_USD` | Urals Crude Oil | barrel | Russian grade — discount reference |
| `SINGAPORE_GASOIL_USD` | Singapore Gasoil | barrel | Asian distillate reference |

> The other ~280 `oil` items are US / India / Japan / Saudi **retail pump prices** — excluded as not relevant to a UK wholesale book.

---

## 🔵 Gas — UK + European hubs (richest, most relevant set)

| Code | Name | Unit | Tier |
|------|------|------|------|
| `NATURAL_GAS_GBP` | UK Natural Gas | therm | **core (UK)** |
| `NATURAL_GAS_TTF_SPOT_EUR` | TTF Natural Gas Spot | MWh | **core (EU benchmark)** |
| `DUTCH_TTF_EUR` | Dutch TTF Natural Gas | MWh | primary EU hub |
| `LNG_NW_EUROPE_EUR` | LNG NW Europe | MWh | LNG import reference |
| `LNG_EU_AVERAGE_EUR` | LNG EU Average | MWh | LNG import reference |
| `NATURAL_GAS_THE_EUR` | THE Natural Gas (Germany) | MWh | cross-border |
| `NATURAL_GAS_PEG_EUR` | PEG Natural Gas (France) | MWh | cross-border (interconnector) |
| `NATURAL_GAS_ZTP_EUR` | ZTP Natural Gas (Belgium) | MWh | cross-border (interconnector) |
| `NATURAL_GAS_PVB_EUR` | PVB Natural Gas (Spain) | MWh | context |

> Also available: CEGH (Austria), CZ VTP, ETF (Estonia), FIN, LTU, LVA-EST national hubs, and `NATURAL_GAS_USD` (US Henry Hub) for global context.

---

## ⚫ Coal — global + US basins

Seaborne benchmarks plus US domestic basins (kept broad per request — not Europe-only):

| Code | Name | Live price | Unit | Region |
|------|------|-----------|------|--------|
| `NEWCASTLE_COAL_USD` | Newcastle Coal (API6) | **$128.60** | metric_ton | Asia-Pacific seaborne benchmark |
| `COAL_USD` | Coal (generic) | **$117.35** | metric_ton | global |
| `COKING_COAL_USD` | Coking Coal | **$237.00** | metric_ton | global (metallurgical) |
| `CAPP_COAL_USD` | Central Appalachian Coal | **$82.00** | short_ton | US East |
| `ILLINOIS_COAL_USD` | Illinois Basin Coal | **$56.00** | short_ton | US Midwest |
| `PRB_COAL_USD` | Powder River Basin Coal | **$14.70** | short_ton | US West (low-sulfur) |

> Annual-average variants also exist: `CAPP_COAL_ANNUAL_USD`, `ILLINOIS_COAL_ANNUAL_USD`, `PRB_COAL_ANNUAL_USD`.
> ⚠️ **No European API2 (ARA) coal** benchmark in the catalog — Newcastle API6 is the closest seaborne reference.
> ⚠️ Discontinued (excluded): `CME_COAL_USD`, `NYMEX_APPALACHIAN_USD`, `NYMEX_WESTERN_RAIL_USD`.
> Note US basins are quoted in **short_ton**; seaborne benchmarks in **metric_ton** — normalize before comparing.

---

## 🚢 Marine fuel — ⚠️ all discontinued

| Code | Name | Status |
|------|------|--------|
| `MGO_05S_USD` | Marine Gas Oil 0.5%S | **Discontinued** |
| `VLSFO_USD` | Very Low Sulfur Fuel Oil | **Discontinued** |
| `HFO_380_USD` | Heavy Fuel Oil 380 CST | **Discontinued** |
| `HFO_180_USD` | Heavy Fuel Oil 180 CST | **Discontinued** |

> All four marine fuels are marked **Discontinued** — no live data. Low relevance to a retail energy supplier; listed so no feed is built on dead codes.

---

## Takeaways for Fuse

- **Gas is the heart of it** — `NATURAL_GAS_GBP` + `NATURAL_GAS_TTF_SPOT_EUR` are the two must-watch feeds.
- **Carbon matters** — `UK_CARBON_GBP` / `EU_CARBON_EUR` are arguably more useful than coal or marine fuel for a UK supplier's cost/compliance model.
- **Coal** — Newcastle API6 for the seaborne benchmark; US basins (CAPP / Illinois / PRB) for domestic/thermal context.
- **Skip marine fuel** — all discontinued.
