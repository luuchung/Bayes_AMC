import matplotlib.pyplot as plt
import numpy as np
def minibatch_weight(batch_idx, num_batches):

    """Calculates Minibatch Weight.

    A formula for calculating the minibatch weight is described in
    section 3.4 of the 'Weight Uncertainty in Neural Networks' paper.
    The weighting decreases as the batch index increases, this is
    because the the first few batches are influenced heavily by
    the complexity cost.

    Parameters
    ----------
    batch_idx : int
        Current batch index.
    num_batches : int
        Total number of batches.

    Returns
    -------
    float
        Current minibatch weight.
    """

    return 2 ** (num_batches - batch_idx) / (2 ** num_batches - batch_idx)

def count_average_prob(out_model):
  mask = (out_model == out_model.max(axis = 1, keepdims = 1)).astype(float)
  result = np.multiply(mask, out_model)
  mean = result.sum(axis = 0)
  count = (result != 0).sum(axis = 0)
  for i in range(len(count)):
    if count[i] != 0:
      mean[i] = mean[i]/count[i]
  return mean

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)
    # plt.title(title,fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, size=12)
    plt.yticks(tick_marks, labels, size=12)
    # np.set_printoptions(precision=2, suppress=True)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i != j:
                text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
            elif i == j:
                if int(np.around(cm[i, j] * 100)) == 100:
                    text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=8,
                                    color='darkorange')
                else:
                    text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10,
                                    color='darkorange')

    plt.tight_layout()
    plt.ylabel('True label',fontdict={'size':8,})
    plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for k in range(0, Y.shape[0]):
        i = list(Y[k, :]).index(1)
        j = int(np.argmax(Y_hat[k, :]))
        conf[i, j] = conf[i, j] + 1

    for i in range(0, n_classes):
        confnorm[i, :] = conf[i, :] / (np.sum(conf[i, :]) + 0.0001)
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong



