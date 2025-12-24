"""
Simple paper-trading backtest for CANBK minute data with technical indicators.

Strategy (swing oriented, long-only):
- Enter when price closes above EMA 20 AND EMA 20 > EMA 50, RSI > 45,
  price above Bollinger lower band. Allocates a fixed fraction of capital.
- Exit when any of:
    * Close falls below EMA 20
    * RSI drops below 40
    * Stop-loss 3% below entry
    * Take-profit 6% above entry

Capital: starts with 200,000 INR.
Commissions/slippage: ignored for simplicity.

Usage:
    python backtest_strategy.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Dict, List, Optional


DATA_FILE = Path("outputs/CANBK_NSE_2763265_indicators.csv")
INITIAL_CAPITAL = 200_000.0
ALLOC_FRACTION = 0.25  # allocate 25% of current capital per trade
STOP_LOSS_PCT = 0.02   # tighter stop to cut losers quickly
TAKE_PROFIT_PCT = 0.08  # wider take-profit for better R:R
COOLDOWN_BARS = 5       # wait bars after an exit before re-entering


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi14: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema_8: float
    ema_20: float
    ema_50: float
    ema_100: float
    ema_200: float


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_bars: int


def read_bars(path: Path) -> List[Bar]:
    bars: List[Bar] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                Bar(
                    date=row["date"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    rsi14=float(row["rsi14"]),
                    bb_mid=float(row["bb_mid"]),
                    bb_upper=float(row["bb_upper"]),
                    bb_lower=float(row["bb_lower"]),
                    macd=float(row["macd"]),
                    macd_signal=float(row["macd_signal"]),
                    macd_hist=float(row["macd_hist"]),
                    ema_8=float(row["ema_8"]),
                    ema_20=float(row["ema_20"]),
                    ema_50=float(row["ema_50"]),
                    ema_100=float(row["ema_100"]),
                    ema_200=float(row["ema_200"]),
                )
            )
    return bars


def max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def backtest(bars: List[Bar]) -> Dict[str, object]:
    capital = INITIAL_CAPITAL
    position_shares = 0
    entry_price = 0.0
    entry_date = ""
    entry_idx: Optional[int] = None
    cooldown = 0
    trades: List[Trade] = []
    equity: List[float] = []

    for idx, bar in enumerate(bars):
        price = bar.close
        # Manage open position first
        if position_shares > 0:
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TAKE_PROFIT_PCT)
            exit_signal = (
                price < bar.ema_20
                or bar.rsi14 < 45
                or bar.macd < bar.macd_signal
                or price <= stop_price
                or price >= target_price
            )
            if exit_signal:
                proceeds = position_shares * price
                capital += proceeds
                pnl = proceeds - (position_shares * entry_price)
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        exit_date=bar.date,
                        entry_price=entry_price,
                        exit_price=price,
                        shares=position_shares,
                        pnl=pnl,
                        pnl_pct=(price - entry_price) / entry_price,
                        hold_bars=idx - entry_idx if entry_idx is not None else 0,
                    )
                )
                position_shares = 0
                entry_price = 0.0
                entry_date = ""
                entry_idx = None
                cooldown = COOLDOWN_BARS

        # Look for entry if flat
        if position_shares == 0 and cooldown == 0:
            entry_cond = (
                price > bar.ema_20
                and bar.ema_20 > bar.ema_50
                and bar.ema_8 > bar.ema_20
                and bar.rsi14 > 52
                and bar.rsi14 < 75
                and bar.macd > bar.macd_signal
                and bar.macd_hist > 0
                and price > bar.bb_lower
            )
            if entry_cond and capital > 0:
                allocation = capital * ALLOC_FRACTION
                shares = floor(allocation / price)
                if shares > 0:
                    cost = shares * price
                    capital -= cost
                    position_shares = shares
                    entry_price = price
                    entry_date = bar.date
                    entry_idx = idx

        # Track equity (mark-to-market)
        mark_to_market = capital + position_shares * price
        equity.append(mark_to_market)

        # Step cooldown counter
        if cooldown > 0 and position_shares == 0:
            cooldown -= 1

    # Close any open position at last bar price
    if position_shares > 0 and bars:
        last_bar = bars[-1]
        price = last_bar.close
        proceeds = position_shares * price
        capital += proceeds
        pnl = proceeds - (position_shares * entry_price)
        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=last_bar.date,
                entry_price=entry_price,
                exit_price=price,
                shares=position_shares,
                pnl=pnl,
                pnl_pct=(price - entry_price) / entry_price,
                hold_bars=len(bars) - 1 - entry_idx if entry_idx is not None else 0,
            )
        )
        equity[-1] = capital  # last point reflects realized value

    return {
        "trades": trades,
        "equity": equity,
        "final_equity": capital,
    }


def summarize_results(result: Dict[str, object]) -> None:
    trades: List[Trade] = result["trades"]  # type: ignore
    equity: List[float] = result["equity"]  # type: ignore
    final_equity: float = result["final_equity"]  # type: ignore

    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    mdd = max_drawdown(equity) * 100 if equity else 0.0

    print(f"Initial capital : {INITIAL_CAPITAL:,.2f} INR")
    print(f"Final equity    : {final_equity:,.2f} INR")
    print(f"Total return    : {total_return:.2f}%")
    print(f"Trades taken    : {len(trades)}")
    print(f"Win rate        : {win_rate:.2f}%")
    print(f"Avg win         : {avg_win:,.2f}")
    print(f"Avg loss        : {avg_loss:,.2f}")
    print(f"Max drawdown    : {mdd:.2f}%")

    print("\nRecent trades (last 5):")
    for t in trades[-5:]:
        print(
            f"{t.entry_date} -> {t.exit_date} | "
            f"{t.shares} sh | {t.entry_price:.2f} -> {t.exit_price:.2f} | "
            f"PnL {t.pnl:,.0f} ({t.pnl_pct*100:.2f}%)"
        )


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Indicator file not found: {DATA_FILE}")

    bars = read_bars(DATA_FILE)
    result = backtest(bars)
    summarize_results(result)


if __name__ == "__main__":
    main()

