import matplotlib.pyplot as plt
with open('data.txt') as f:
    lines = f.read().split('\n')
    lines = [l.split('\t') for l in lines]
    condition = [float(l[3]) for l in lines[:10000]]
    plt.hist(condition, bins = [0.01*i for i in range(100)])
    plt.show()
