from pathlib import Path
import argparse, json, hashlib, platform, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from model import clean, estimate, library
from engine import backtest, metrics
CODES = ['000300.XSHG', '000001.XSHG', '399006.XSHE']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def job(task):
    code, i, day, px, seed = task
    row = dict(time=str(day), **estimate(np.diff(np.log(clean(px))) * 100, 1000, seed))
    target = Path('fit_progress')
    target.mkdir(exist_ok=True)
    (target / f'{code}_{i}.json').write_text(json.dumps(dict(i=i, **row), default=lambda v: v.item()))
    return (code, i, row)

def prepare(g):
    c = g.close
    change = c.diff()
    er = ((c - c.shift(30)).abs() / change.abs().rolling(30).sum()).clip(0, 1).fillna(0)
    win = (250 - 229 * er).round().astype(int)
    mw = (20 - 15 * er).round().astype(int)
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rsi = (100 * gain / (gain + loss)).mask(gain + loss == 0, 50)
    tr = pd.concat([g.high - g.low, (g.high - c.shift()).abs(), (g.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).apply(lambda x: np.clip(x, *np.quantile(x, [0.05, 0.95])).mean(), raw=True)
    return pd.DataFrame(dict(time=g.time, window=win, ma_window=mw, rsi=rsi, atr=atr, ma=[c.iloc[max(0, i - int(mw.iloc[i]) + 1):i + 1].mean() for i in range(len(c))]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--daily', required=True)
    p.add_argument('--minute', required=True)
    p.add_argument('--out', default='results/full')
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    library()
    d = pd.read_excel(a.daily) if a.daily.endswith('xlsx') else pd.read_csv(a.daily)
    m = pd.read_csv(a.minute)
    d.time = pd.to_datetime(d.time)
    m.time = pd.to_datetime(m.time)
    d = d[d.time <= '2026-05-31']
    m = m[m.time < '2026-06-01']
    m['day'] = m.time.dt.normalize()
    if d.duplicated(['code', 'time']).any() or m.duplicated(['code', 'time']).any():
        raise ValueError('Duplicate bars')
    if set(CODES) - set(d.code):
        raise ValueError('Missing codes')
    for frame in [d, m]:
        if not np.isfinite(frame[['open', 'high', 'low', 'close']]).all().all() or (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
            raise ValueError('Invalid prices')
    daily = {code: d[d.code == code].sort_values('time').reset_index(drop=True) for code in CODES}
    signals = {code: prepare(g) for code, g in daily.items()}
    tasks = []
    result = {code: [] for code in CODES}
    for code, g in daily.items():
        for i, r in signals[code].iterrows():
            n = int(r.window)
            if i >= max(n, 30):
                tasks.append((code, i, r.time, g.close.iloc[i - n:i + 1].to_numpy(), 20260610 + CODES.index(code) * 100000 + i))
    manifest = dict(status='RUNNING', B=1000, bootstrap='recursive standardized residual AR-GARCH; joint refit each sample', expected_daily_fits=len(tasks), daily_sha256=sha(a.daily), minute_sha256=sha(a.minute), start='2020-01-01', end='2026-05-31', versions={'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}, source_hashes={str(f): sha(f) for f in list(Path('.').glob('*.py')) + list(Path('native').glob('*.cpp'))})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for count, f in enumerate(as_completed([pool.submit(job, x) for x in tasks]), 1):
            try:
                code, i, row = f.result()
            except Exception as exc:
                print('FAILED', repr(exc), flush=True)
                raise
            result[code].append(dict(i=i, **row))
            if count % 100 == 0:
                print(f'FULL {count}/{len(tasks)} elapsed={time.time() - t0:.0f}s', flush=True)
            if count % 500 == 0:
                for code in CODES:
                    pd.DataFrame(result[code]).to_csv(out / f'{code}_checkpoint.csv', index=False)
    summary = []
    years = []
    audit = []
    for code, g in daily.items():
        fits = pd.DataFrame(result[code]).set_index('i')
        s = signals[code].join(fits.drop(columns='time'))
        s['state'] = 0
        s['signal'] = 0
        hist = []
        sig_hist = []
        streak = 0
        for i, r in s.iterrows():
            if not np.isfinite(r.get('tsi', np.nan)) or r.get('fit_ok') != True:
                streak = 0
                continue
            hist.append(abs(r.tsi))
            sig_hist.append(r.sigma)
            tp = np.mean(np.array(hist) <= abs(r.tsi))
            sp = np.mean(np.array(sig_hist) <= r.sigma)
            good = r.bootstrap_valid >= 950 and ((r.ci_low > 0 or r.ci_high < 0) or tp > 0.5) and (r.rho > 0) and (r.rho_t > 1.5 or sp < 0.95)
            streak = streak + 1 if good else 0
            state = int(np.sign(r.tsi)) if streak >= 10 else 0
            s.loc[i, 'tsi_percentile'] = tp
            s.loc[i, 'sigma_percentile'] = sp
            s.loc[i, 'state'] = state
            s.loc[i, 'signal'] = (1 if state > 0 and g.close.iloc[i] > r.ma else -1 if state < 0 and g.close.iloc[i] < r.ma else 0) if state else 1 if r.rsi < 30 else -1 if r.rsi > 70 else 0
        s.to_csv(out / f'{code}_signals.csv', index=False)
        mm = m[m.code == code].sort_values('time')
        groups = dict(tuple(mm.groupby('day')))
        ag = mm.groupby('day').agg(minute_open=('open', 'first'), minute_close=('close', 'last'), bars=('open', 'size'))
        check = g.set_index('time').join(ag)
        check['open_relative_difference'] = (check.open - check.minute_open).abs() / check.open
        check.to_csv(out / f'{code}_data_check.csv')
        for variant in ['full', 'always_trend', 'always_range', 'no_atr']:
            nav, trades = backtest(g, s, groups, variant, 4.5, '2020-01-01', '2026-05-31', 0.002)
            nav.to_csv(out / f'{code}_{variant}_nav.csv', index=False)
            trades.to_csv(out / f'{code}_{variant}_trades.csv', index=False)
            summary.append(dict(code=code, strategy=variant, **metrics(nav.nav, trades)))
            for year, part in nav.groupby(nav.time.dt.year):
                prev = nav.loc[nav.time < part.time.min(), 'nav']
                base = prev.iloc[-1] if len(prev) else 1.0
                years.append(dict(code=code, strategy=variant, year=year, **metrics(part.nav / base)))
        base = g[g.time >= '2020-01-01']
        summary.append(dict(code=code, strategy='benchmark', **metrics(base.close / base.open.iloc[0])))
        audit.append(dict(code=code, fits=len(fits), point_failures=int((~fits.fit_ok).sum()), bootstrap_requested=int(fits.bootstrap_requested.sum()), bootstrap_valid=int(fits.bootstrap_valid.sum()), insufficient_days=int((fits.bootstrap_valid < 950).sum())))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(years).to_csv(out / 'yearly.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'fit_audit.csv', index=False)
    manifest.update(status='COMPLETED', elapsed_seconds=time.time() - t0, completed_daily_fits=sum((len(x) for x in result.values())))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
