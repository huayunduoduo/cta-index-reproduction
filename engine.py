import numpy as np
import pandas as pd

def backtest(d, s, minutes, variant, k, start, end, cost=0.002):
    z = d.merge(s, on='time', suffixes=('', '_s'))
    cash = 1.0
    qty = 0.0
    entry = None
    pending = 0
    exit_pending = False
    peak = trough = stop = None
    records = []
    trades = []
    selected = z[(z.time >= start) & (z.time <= end)]
    for ix, r in selected.iterrows():
        day = r.time
        was = qty != 0
        op = r.open
        fee = 0.0
        reason = None
        fill = None
        if was:
            if exit_pending:
                fill = op
                reason = 'rsi_or_opposite_signal_next_open'
            elif entry['kind'] == 'trend' and variant != 'no_atr' and (stop is not None):
                if qty > 0 and op <= stop or (qty < 0 and op >= stop):
                    fill = op
                    reason = 'gap_stop'
                else:
                    bars = minutes.get(day)
                    if bars is None:
                        raise ValueError(f'Minute data missing: {day}')
                    hit = bars[bars.low <= stop] if qty > 0 else bars[bars.high >= stop]
                    if len(hit):
                        b = hit.iloc[0]
                        fill = min(b.open, stop) if qty > 0 else max(b.open, stop)
                        reason = 'minute_stop'
        elif pending:
            nav = cash
            effective = op * (1 + pending * cost / 2)
            qty = pending * nav / op
            cash -= qty * effective + abs(qty * effective) * cost / 2
            entry = dict(date=day, price=op, direction=pending, kind=pending_kind, equity=nav)
            peak = op
            trough = op
            stop = None
        if fill is not None:
            effective = fill * (1 - np.sign(qty) * cost / 2)
            cash += qty * effective - abs(qty * effective) * cost / 2
            trades.append(dict(entry_date=entry['date'], exit_date=day, direction=entry['direction'], kind=entry['kind'], entry_price=entry['price'], exit_price=fill, return_net=cash / entry['equity'] - 1, reason=reason))
            qty = 0.0
            entry = None
            stop = None
        equity = cash + qty * r.close
        exit_pending = False
        pending = 0
        if qty:
            if entry['kind'] == 'trend':
                peak = max(peak, r.high)
                trough = min(trough, r.low)
                if np.isfinite(r.atr):
                    new = peak - k * r.atr if qty > 0 else trough + k * r.atr
                    stop = new if stop is None else max(stop, new) if qty > 0 else min(stop, new)
                if variant == 'no_atr':
                    exit_pending = qty > 0 and r.signal < 0 or (qty < 0 and r.signal > 0)
            else:
                exit_pending = qty > 0 and r.rsi > 40 or (qty < 0 and r.rsi < 60)
        else:
            signal = r.signal
            kind = 'trend' if r.state else 'range'
            if variant == 'always_trend':
                signal = 1 if r.close > r.get('ma', np.nan) else -1 if r.close < r.get('ma', np.nan) else 0
                kind = 'trend'
            elif variant == 'always_range':
                signal = 1 if r.rsi < 30 else -1 if r.rsi > 70 else 0
                kind = 'range'
            pending = signal
            pending_kind = kind
        records.append(dict(time=day, nav=equity, position=np.sign(qty), stop=stop, state=r.state, close=r.close))
    if qty:
        terminal = selected.iloc[-1].close * (1 - np.sign(qty) * cost / 2)
        cash += qty * terminal - abs(qty * terminal) * cost / 2
        records[-1]['nav'] = cash
        trades.append(dict(entry_date=entry['date'], exit_date=records[-1]['time'], direction=entry['direction'], kind=entry['kind'], entry_price=entry['price'], exit_price=selected.iloc[-1].close, return_net=cash / entry['equity'] - 1, reason='terminal_liquidation'))
    return (pd.DataFrame(records), pd.DataFrame(trades))

def metrics(nav, trades=None):
    v = np.r_[1.0, np.asarray(nav)]
    r = v[1:] / v[:-1] - 1
    n = len(nav)
    ann = v[-1] ** (252 / n) - 1
    dd = np.min(v / np.maximum.accumulate(v) - 1)
    vol = np.std(r, ddof=1) * np.sqrt(252)
    out = dict(annual_return=ann, max_drawdown=dd, calmar=ann / abs(dd) if dd else np.nan, sharpe=np.mean(r) / np.std(r, ddof=1) * np.sqrt(252) if vol else np.nan, annual_volatility=vol, total_return=v[-1] - 1, days=n)
    if trades is not None:
        t = trades.return_net if len(trades) else pd.Series(dtype=float)
        wins = t[t > 0]
        loss = t[t < 0]
        out.update(trades=len(t), win_rate=(t > 0).mean(), payoff_ratio=wins.mean() / abs(loss.mean()) if len(loss) and len(wins) else np.nan)
    return out
