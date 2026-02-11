plt.figure(figsize=(8,6))

colors = ["red","orange","m","#AA55B7",
          "#11FF99","c","#986432","g"]

for k, col in enumerate(colors):
    cluster_data = (y_pred == k)
    plt.scatter(
        X[cluster_data, 0],
        X[cluster_data, 1],
        c=col,
        marker=".",
        s=10
    )

# centroidy
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Centroidy"
)

plt.title("Wynik algorytmu k-means")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()
