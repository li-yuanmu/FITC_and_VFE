
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

data = np.genfromtxt("/Users/lym/Desktop/GP/Study_GPflow/data/regression_1D.csv", delimiter=",")
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

k = gpflow.kernels.RBF()

m = gpflow.models.GPRFITC(data=(X,Y),kernel=k, inducing_variable=X.copy(),mean_function=None)

m.likelihood.variance.assign(0.01)

#训练模型
opt = gpflow.optimizers.Scipy()
def objective_closure():
    return -m.log_marginal_likelihood()
opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)
print(m.inducing_variable)

## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)


## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)


plt.show()