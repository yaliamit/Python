
import amitgroup as ag
import numpy as np

plw = ag.plot.PlottingWindow(figsize=(4,2), subplots=(1,2))
faces = ag.io.load_example('faces')
N = len(faces)
x = 0
while x <= 200 and plw.tick():
    plw.imshow(faces[x%N], subplot=0)
    plw.plot(np.sin(np.arange(x)/10.0), limits=(-1, 1), subplot=1)
    plw.flip()
    x += 1

# Optional (if you want the window to persist and block until user quits)
plw.mainloop()

