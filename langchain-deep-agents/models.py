from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ETF:
    symbol: str
    name: str
    asset_class: str
    aum_millions: Optional[float] = None      # AUM in $M
    fund_flow_pct: Optional[float] = None     # YTD fund flow %
    avg_volume: Optional[int] = None
    price: Optional[float] = None
    one_year_return_pct: Optional[float] = None

    def to_node_attrs(self) -> dict:
        return {
            "type": "etf",
            "name": self.name,
            "asset_class": self.asset_class,
            "aum_millions": self.aum_millions,
            "fund_flow_pct": self.fund_flow_pct,
            "avg_volume": self.avg_volume,
            "price": self.price,
            "one_year_return_pct": self.one_year_return_pct,
        }


@dataclass
class Issuer:
    name: str           # e.g. "Vanguard"
    slug: str           # e.g. "vanguard"  (URL path segment)
    acronym: str        # e.g. "VG"
    url: str
    aum_billions: Optional[float] = None
    etfs: list[ETF] = field(default_factory=list)

    @property
    def equity_etfs(self) -> list[ETF]:
        return [e for e in self.etfs if e.asset_class.lower() == "equity"]

    def to_node_attrs(self) -> dict:
        return {
            "type": "issuer",
            "name": self.name,
            "slug": self.slug,
            "acronym": self.acronym,
            "url": self.url,
            "aum_billions": self.aum_billions,
        }
