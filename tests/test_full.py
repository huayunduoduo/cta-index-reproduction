import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from model import library, estimate
from engine import backtest, metrics

class ModelTests(unittest.TestCase):

    def test_native_exact_gradient(self):
        y = np.ascontiguousarray(np.random.default_rng(5).normal(size=100))
        z = np.array([0.1, 0.1, -2.0, 0.0, 2.0])
        lib = library()
        g = np.zeros(5)
        lib.gradient(z, y, len(y), g)
        num = []
        for j in range(5):
            a = z.copy()
            b = z.copy()
            a[j] += 1e-05
            b[j] -= 1e-05
            num.append((lib.objective(a, y, len(y)) - lib.objective(b, y, len(y))) / 2e-05)
        np.testing.assert_allclose(g, num, rtol=1e-05, atol=1e-05)

    def test_joint_fit_matches_scipy_objective(self):
        y = np.ascontiguousarray(np.random.default_rng(8).normal(size=180))
        o = np.zeros(7)
        z = np.zeros(5)
        lib = library()
        lib.fit_model(y, len(y), o, z)
        v = lib.objective(z, y, len(y))
        r = minimize(lambda x: lib.objective(np.ascontiguousarray(x), y, len(y)), z, method='BFGS')
        self.assertLess(v - r.fun, 0.02)

    def test_full_bootstrap_deterministic(self):
        y = np.random.default_rng(3).normal(size=80)
        a = estimate(y)
        b = estimate(y)
        self.assertEqual(a['bootstrap_requested'], 1000)
        self.assertEqual(a['ci_low'], b['ci_low'])
        self.assertGreaterEqual(a['bootstrap_valid'], 950)

    def test_initial_drawdown(self):
        self.assertAlmostEqual(metrics([0.9, 0.95])['max_drawdown'], -0.1)

    def fixture(self, price):
        n = len(price)
        d = pd.DataFrame({'time': pd.bdate_range('2020-01-01', periods=n), 'open': price, 'close': price, 'high': np.array(price) + 1, 'low': np.array(price) - 1})
        s = pd.DataFrame({'time': d.time, 'signal': [1] * n, 'state': [1] * n, 'atr': [1.0] * n, 'rsi': [50.0] * n, 'ma': np.array(price) - 1})
        m = {r.time: pd.DataFrame({'open': [r.open], 'high': [r.high], 'low': [r.low]}) for r in d.itertuples()}
        return (d, s, m)

    def test_flat_price_roundtrip_cost(self):
        d, s, m = self.fixture([100, 100, 100])
        n, t = backtest(d, s, m, 'full', 4.5, '2020-01-01', '2020-01-05', 0.002)
        self.assertAlmostEqual(n.nav.iloc[-1], 0.996)

    def test_no_reentry_at_past_open(self):
        d, s, m = self.fixture([100, 100, 80, 80])
        n, t = backtest(d, s, m, 'full', 4.5, '2020-01-01', '2020-01-06')
        self.assertEqual(t.iloc[0].exit_price, 80)
        self.assertNotEqual(t.iloc[1].entry_date, t.iloc[0].exit_date)

    def test_zero_intraday_return_no_overnight_profit(self):
        d, s, m = self.fixture([90, 100, 100])
        n, t = backtest(d, s, m, 'full', 4.5, '2020-01-01', '2020-01-05', 0)
        self.assertEqual(n.nav.iloc[-1], 1.0)

    def test_future_price_cannot_change_past_nav(self):
        d, s, m = self.fixture([100, 100, 101, 102, 103])
        a, _ = backtest(d, s, m, 'full', 4.5, '2020-01-01', '2020-01-07')
        d.loc[4, ['open', 'close', 'high', 'low']] = 500
        b, _ = backtest(d, s, m, 'full', 4.5, '2020-01-01', '2020-01-07')
        np.testing.assert_allclose(a.nav.iloc[:4], b.nav.iloc[:4])
if __name__ == '__main__':
    unittest.main()
