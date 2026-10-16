"""Type stubs for the compiled ``probatum._probatum`` extension: ``Engine``, ``Batch``,
``State``, ``periods_per_year`` and ``run``.
Hand-maintained; keep in sync with ``crates/probatum-py/src/lib.rs``. Order, fill and
event records come back as plain dicts, typed here as ``dict[str, Any]``.
"""

from typing import Any, Final, Optional

import numpy as np

class State:
    tick_index: Final[int]
    base: Final[float]
    quote: Final[float]
    spot_cash: Final[float]
    margin_cash: Final[float]
    position: Final[float]
    avg_entry: Final[float]
    equity: Final[float]
    mark_price: Final[float]
    bar_open: Final[float]
    bar_high: Final[float]
    bar_low: Final[float]
    bar_close: Final[float]
    bar_volume: Final[float]
    realized_pnl: Final[float]
    unrealized_pnl: Final[float]
    funding_paid: Final[float]
    @property
    def open_orders(self) -> list[dict[str, Any]]: ...
    def __getitem__(self, key: str) -> Any: ...

class Engine:
    def __init__(
        self,
        candles: np.ndarray,
        market: str = ...,
        quote: float | dict[str, float] = ...,
        fee_taker: float = ...,
        fee_maker: float = ...,
        slippage_bps: float = ...,
        budget: float = ...,
        window: int = ...,
        max_open_orders: int = ...,
        report: bool = ...,
        leverage: float = ...,
        impact: float = ...,
        impact_model: str = ...,
        spread: Optional[float | str] = ...,
        limit: str = ...,
        sizes: bool = ...,
        funding_rate: float = ...,
        funding_interval: int = ...,
        times: Optional[np.ndarray] = ...,
        funding_hours: float = ...,
        funding_times: Optional[np.ndarray] = ...,
        funding_rates: Optional[np.ndarray] = ...,
        funding_instruments: Optional[list[Optional[int]]] = ...,
        tiers: Optional[list[tuple[float, float, float, float]]] = ...,
        tiers_by: str = ...,
        lot: float = ...,
        tick: float = ...,
        sigfigs: Optional[int] = ...,
        decimals: Optional[int] = ...,
        min_notional: float = ...,
        max_qty: Optional[float] = ...,
        reduce_only_exempt: bool = ...,
        instruments: Optional[list[dict[str, Any]]] = ...,
        intrabar: str = ...,
        seed: Optional[int] = ...,
        contract: Optional[float] = ...,
        liquidation: str = ...,
        backstop: Optional[float] = ...,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def data(self) -> np.ndarray: ...
    @property
    def opens(self) -> np.ndarray: ...
    @property
    def highs(self) -> np.ndarray: ...
    @property
    def lows(self) -> np.ndarray: ...
    @property
    def closes(self) -> np.ndarray: ...
    @property
    def volumes(self) -> np.ndarray: ...
    def reset(self) -> State: ...
    def step(self) -> State: ...
    def done(self) -> bool: ...
    def periods_per_year(self) -> Optional[float]: ...
    def observation(self, lookback: int) -> np.ndarray: ...
    def market_buy(self, size: float) -> Optional[int]: ...
    def market_sell(self, size: float) -> Optional[int]: ...
    def limit_buy(self, size: float, price: float) -> Optional[int]: ...
    def limit_sell(self, size: float, price: float) -> Optional[int]: ...
    def stop(
        self, side: str, size: float, trigger: float, reduce_only: bool = ...
    ) -> Optional[int]: ...
    def bracket(
        self, stop: Optional[float] = ..., take_profit: Optional[float] = ...
    ) -> tuple[Optional[int], Optional[int]]: ...
    def order(
        self,
        side: str,
        size: float,
        type: str = ...,
        price: Optional[float] = ...,
        trigger: Optional[float] = ...,
        reduce_only: bool = ...,
        post_only: bool = ...,
        tif: str = ...,
    ) -> Optional[int]: ...
    def close(self) -> Optional[int]: ...
    def cancel(self, order_id: int) -> bool: ...
    def cancel_all(self) -> int: ...
    def transfer(self, amount: float, to: str) -> None: ...
    def target(
        self,
        weights: float | list[float] | np.ndarray | dict[int, float],
        tolerance: float = ...,
        transfer: bool = ...,
    ) -> list[int]: ...
    def replace(
        self,
        order_id: int,
        size: Optional[float] = ...,
        price: Optional[float] = ...,
        trigger: Optional[float] = ...,
    ) -> Optional[int]: ...
    def is_bust(self) -> bool: ...
    def num_fills(self) -> int: ...
    def qty_from_weight(self, fraction: float) -> float: ...
    def qty_from_quote(self, cash: float) -> float: ...
    def run(self, strategy: Any) -> State: ...
    def stats(
        self, periods_per_year: float = ..., risk_free: float = ...
    ) -> Optional[dict[str, Any]]: ...
    def equity_curve(self) -> Optional[np.ndarray]: ...
    def trades(self) -> Optional[list[dict[str, Any]]]: ...
    def ledger(self) -> Optional[dict[str, Any]]: ...
    def fills(self) -> Optional[list[dict[str, Any]]]: ...
    def order_log(self) -> Optional[list[dict[str, Any]]]: ...

class Batch:
    @property
    def num_envs(self) -> int: ...
    def __init__(
        self,
        candles: np.ndarray,
        num_envs: int,
        market: str = ...,
        quote: float | dict[str, float] = ...,
        fee_taker: float = ...,
        fee_maker: float = ...,
        slippage_bps: float = ...,
        budget: float = ...,
        window: int = ...,
        max_open_orders: int = ...,
        report: bool = ...,
        leverage: float = ...,
        impact: float = ...,
        impact_model: str = ...,
        spread: Optional[float | str] = ...,
        limit: str = ...,
        sizes: bool = ...,
        funding_rate: float = ...,
        funding_interval: int = ...,
        fee_taker_per_env: Optional[np.ndarray] = ...,
        fee_maker_per_env: Optional[np.ndarray] = ...,
        slippage_bps_per_env: Optional[np.ndarray] = ...,
        impact_per_env: Optional[np.ndarray] = ...,
        times: Optional[np.ndarray] = ...,
        funding_hours: float = ...,
        funding_times: Optional[np.ndarray] = ...,
        funding_rates: Optional[np.ndarray] = ...,
        funding_instruments: Optional[list[Optional[int]]] = ...,
        tiers: Optional[list[tuple[float, float, float, float]]] = ...,
        tiers_by: str = ...,
        lot: float = ...,
        tick: float = ...,
        sigfigs: Optional[int] = ...,
        decimals: Optional[int] = ...,
        min_notional: float = ...,
        max_qty: Optional[float] = ...,
        reduce_only_exempt: bool = ...,
        instruments: Optional[list[dict[str, Any]]] = ...,
        intrabar: str = ...,
        seed: Optional[int] = ...,
        contract: Optional[float] = ...,
        liquidation: str = ...,
        backstop: Optional[float] = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def done(self) -> bool: ...
    def periods_per_year(self) -> Optional[float]: ...
    def reset_all(self) -> list[State]: ...
    def step_all(self, actions: Optional[np.ndarray] = ...) -> list[State]: ...
    def advance(self, actions: Optional[np.ndarray] = ...) -> None: ...
    def target(
        self, weights: np.ndarray, tolerance: float = ..., transfer: bool = ...
    ) -> None: ...
    def records(self, env: int) -> dict[str, Any]: ...
    def observe(self, window: int) -> np.ndarray: ...
    def set_features(self, features: np.ndarray) -> None: ...
    def observe_features(self, window: int) -> np.ndarray: ...
    def equities(self) -> np.ndarray: ...
    def dones(self) -> np.ndarray: ...
    def busts(self) -> np.ndarray: ...
    def snapshot(self) -> dict[str, np.ndarray]: ...
    def reset_at_each(self, offsets: np.ndarray) -> None: ...
    def reset_masked(self, mask: np.ndarray, offsets: np.ndarray) -> None: ...

def periods_per_year(times: np.ndarray) -> Optional[float]: ...
def run(
    candles: np.ndarray,
    venue: Optional[dict[str, Any]],
    script: dict[int, list[tuple[str, dict[str, Any]]]],
    times: Optional[np.ndarray] = ...,
) -> dict[str, Any]: ...
def arrow_columns(data: Any) -> list[tuple[str, str, Optional[np.ndarray], str]]: ...
