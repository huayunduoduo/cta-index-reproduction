from pathlib import Path
import pandas as pd, numpy as np, json, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).parent
R = ROOT / 'results/full'
manifest = json.loads((R / 'manifest.json').read_text())
if manifest['status'] != 'COMPLETED':
    raise RuntimeError('Full computation not complete')
s = pd.read_csv(R / 'summary.csv')
audit = pd.read_csv(R / 'fit_audit.csv')
years = pd.read_csv(R / 'yearly.csv')
audit['bootstrap_attempts'] = [int(pd.read_csv(R / f'{code}_signals.csv').bootstrap_attempts.sum()) for code in audit.code]
audit.to_csv(R / 'fit_audit.csv', index=False)
html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>三指数完整CTA复现</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:35px auto;padding:20px;color:#20314b}h1,h2{color:#154d7d}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:9px;border-bottom:1px solid #ddd}th{background:#eef4fc}img{width:100%}.note{padding:18px;background:#fff2d5}</style><h1>三指数CTA：完整计算结果</h1><p class="note">与原文ETF交易对象不同，不能宣称精确复现原表数字。每个有效日期执行1000条有效递归AR-GARCH Bootstrap联合重估；没有固定方差Bootstrap近似。数值不收敛样本会补抽，因此attempts可能超过requested；若某日期失败比例较高，置信区间存在数值筛选敏感性，不能把有效条数等同于无估计风险。具体假设见docs/rules.html。2020年1月至2026年5月；之前历史仅预热。</p>', '<h2>计算完成证据</h2>', audit.to_html(index=False, border=0), f"<p>已完成日期估计{manifest['completed_daily_fits']}个，完整运行耗时{manifest['elapsed_seconds']:.1f}秒；种子为代码索引及日期序号确定，所有源与原始数据哈希保存于manifest.json。随机数分布实现及数值库跨平台不保证逐位一致。</p>", '<h2>净绩效</h2>', s.to_html(index=False, border=0, float_format=lambda x: f'{x:.5f}'), '<p>annual_return/max_drawdown等为小数，0.10代表10%；calmar为年化收益除以最大回撤绝对值。Sharpe未扣无风险收益。交易数包含末日强平；指数多空为理论账户，没有借券、利息、融资与指数可直接买卖的承诺。</p>']
for code in s.code.unique():
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for strategy in ['full', 'always_trend', 'always_range', 'no_atr']:
        z = pd.read_csv(R / f'{code}_{strategy}_nav.csv', parse_dates=['time'])
        v = z.nav
        ax[0].plot(z.time, v, label=strategy, lw=1)
        ax[1].plot(z.time, np.array(v) / np.maximum.accumulate(np.r_[1, np.array(v)])[1:] - 1, lw=1)
    z = pd.read_csv(R / f'{code}_full_nav.csv', parse_dates=['time'])
    g = pd.read_csv(R / f'{code}_data_check.csv', parse_dates=['time'])
    g = g[g.time >= z.time.min()]
    v = g.close / g.open.iloc[0]
    ax[0].plot(g.time, v, label='benchmark', lw=1)
    ax[1].plot(g.time, np.array(v) / np.maximum.accumulate(np.r_[1, np.array(v)])[1:] - 1, lw=1)
    ax[0].legend(ncol=3)
    ax[0].set_title(code)
    ax[0].set_ylabel('NAV')
    ax[1].set_ylabel('Drawdown')
    fig.tight_layout()
    f = R / f'{code}.png'
    fig.savefig(f, dpi=140)
    plt.close(fig)
    html.extend([f'<h2>{code}</h2>', f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}">', years[years.code == code].to_html(index=False, border=0, float_format=lambda x: f'{x:.4f}')])
html.extend(['<h2>数据边界与结果解释</h2><p>输入包含三个股票指数；原文基准为沪深300ETF、黄金ETF、国债ETF。指数数据2020-04-20存在沪深300日开盘与首分钟开盘约2.54%的差异，本版未擅自修正。以日开盘成交、分钟高低触发止损，需向提供方核实该差异。每个日期的对账保存为data_check.csv。</p><p>无ATR对照的离场、ER线性映射、Bootstrap重采样方法、rho标准误、MAD阈值、ATR截断窗口和RSI退出时点等原文未完全说明，本版固定假设且不按绩效择优。1000次完整估计并不解决模型设定、数据或披露不足问题。若某对照表现更好，应保留该结论，不能反向改变规则以匹配研报收益。</p><h2>运行与GitHub</h2><p>README包含完整命令；原始数据不随仓库再分发。CI仅运行合成账本、梯度和完整Bootstrap的小样本测试，不伪装全数据运行。实际全量运行以当前结果和清单为证。未加宏观、VaR/ES或多资产配置增量。</p></html>'])
(R / 'report.html').write_text(''.join(html), encoding='utf-8')
print(R / 'report.html')
