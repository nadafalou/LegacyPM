exec(open(__file__.replace('faint.py','eval_e2e.py')).read().split("print(f\"\\nFraction")[0])
m0 = P['ivar'] <= 1e4
f = lambda k, m: PIX * np.mean([rsig(P[f'{k}_x'][m]), rsig(P[f'{k}_y'][m])])
err = lambda m: PIX * 1.4826*np.median(np.abs(P['none_x'][m]-np.median(P['none_x'][m]))) / np.sqrt(2*m.sum())
print('faint (sigma>10 mas) Gaia-star measurements; robust sigma (mas) by stage: DCR -> +RM(new) -> +LM(new)')
for name, m in {'all faint': m0,
                'colour in fit range (-0.5..4)': m0 & (P['c'] > -0.5) & (P['c'] < 4),
                'colour outside fit range': m0 & ((P['c'] <= -0.5) | (P['c'] >= 4)),
                'faint, 10-20 mas': m0 & (P['ivar'] > 1/0.02**2),
                'faint, >20 mas': m0 & (P['ivar'] <= 1/0.02**2)}.items():
    if m.sum() < 50: print(f'  {name}: N={m.sum()}'); continue
    print(f"  {name:32s} N={m.sum():6d}  DCR {f('none',m):6.2f}  +RM {f('rm_new',m):6.2f}  +LM {f('r_new',m):6.2f}   (stat. err of each ~{err(m):.2f})")
