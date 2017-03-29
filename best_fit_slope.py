from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


xs = np.array([1, 2, 3, 4, 5, 6],dtype=float)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=float)


def best_fit_slope_and_intercept(xs,ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs*ys) ) / ( (mean(xs)**2) - mean(xs**2) ))  #xs^2 square
    b = mean(ys) - m*mean(xs)
    return m, b

m,b  = best_fit_slope_and_intercept(xs,ys)
print(m, b)


regression_line = [(m*x)+b for x in xs]
predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs,regression_line)
plt.show()