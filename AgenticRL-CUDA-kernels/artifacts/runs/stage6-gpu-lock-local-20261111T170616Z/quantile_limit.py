import torch,time,json
size=2**24+1
x=torch.zeros(size)
started=time.monotonic()
try:
 torch.quantile(x,.99)
except RuntimeError as exc:
 print(json.dumps({'torch':torch.__version__,'elements':size,'error':str(exc),'elapsed_s':time.monotonic()-started}),flush=True)
else:
 print(json.dumps({'torch':torch.__version__,'elements':size,'error':None}),flush=True)
