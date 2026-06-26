export const MODEL_NEMOTRON = 'nvidia/nemotron-3-ultra-550b-a55b:free';
export const MODEL_LAGUNA   = 'poolside/laguna-m.1:free';
export const MODEL_GEMINI   = 'google/gemini-2.5-pro';
export const MODEL          = MODEL_GEMINI;
export const WEAVE_PROJECT = 'elenamylocuda-gemma/Financial MP';
export const VIX_SYMBOL    = '^VIX';

// Single source of truth for all 9 tracked symbols.
// sym    — FMP/Yahoo Finance ticker
// label  — human-readable display name
// region — used for regional grouping in reports
export const INDICES = [
  { sym: '^VIX',      label: 'VIX (Fear Index)', region: 'US'     },
  { sym: '^GSPC',     label: 'S&P 500',           region: 'US'     },
  { sym: '^DJI',      label: 'Dow Jones',          region: 'US'     },
  { sym: '^IXIC',     label: 'NASDAQ Comp.',       region: 'US'     },
  { sym: '^RUT',      label: 'Russell 2000',       region: 'US'     },
  { sym: '^FTSE',     label: 'FTSE 100',           region: 'Europe' },
  { sym: '^STOXX50E', label: 'Euro STOXX 50',      region: 'Europe' },
  { sym: '^N225',     label: 'Nikkei 225',         region: 'Asia'   },
  { sym: '^HSI',      label: 'Hang Seng',          region: 'Asia'   },
];
