from WindPy import w
w.start()
AL = w.wsi("ALI.CMX", "open,high,low,close,amt,volume,oi", "2025-01-01 09:00:00", "2026-04-03 17:12:50",usedf=True )
AL[1].dropna().to_csv('AL.csv')
#z= w.wsi("SPTAUUSDOZ.IDC", "open,high,low,close,amt,volume,oi", "2025-01-01 09:00:00", "2026-04-03 17:12:50", usedf=True)
#z[1].dropna().to_csv('ldjx_1min.csv')
