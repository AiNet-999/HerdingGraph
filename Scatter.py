import numpy as np
import matplotlib.pyplot as plt

tickers = ["MMM", "AOS", "ABT", "ABBV", "ACN"]
num_plot_stocks = 5

plt.figure(figsize=(5, 4))

colors = ['blue', 'red', 'green', 'purple', 'orange']

for stock_id in range(num_plot_stocks):
    plt.scatter(
        y_true_inv[:, stock_id],
        y_pred_inv[:, stock_id],
        alpha=0.6,
        s=45,
        color=colors[stock_id],
        label=tickers[stock_id]
    )

min_val = np.min(y_true_inv[:, :num_plot_stocks])
max_val = np.max(y_true_inv[:, :num_plot_stocks])

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--',
    linewidth=2
)

plt.xlabel("True Price", fontsize=14)
plt.ylabel("Predicted Price", fontsize=14)
plt.title("True vs Predicted Stock Prices", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("1.png")
plt.show()