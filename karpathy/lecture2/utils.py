import matplotlib.pyplot as plt

def plot_bigram_tensor(N, itos):
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')
    for i in range(len(N)):
        for j in range(len(N)):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.show()