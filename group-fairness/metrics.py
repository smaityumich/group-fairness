import tensorflow as tf
import utils
import numpy as np

def predict(x, model, threshold = 0.5):
    logits = model(x)
    prob = utils.logit_to_probability(logits)[:, 1]
    prob = prob.numpy()
    return prob > threshold


def metrics(x, grouped_x, seed, lr, wlr, epsilon, w_reg, l2_reg):
    filename = f'saved-models/seed-{seed}-lr-{lr}-wlr-{wlr}-epsilon-{epsilon}-w_reg-{w_reg}-l2_reg-{l2_reg}'
    model = tf.keras.models.load_model(filename)

    # DD-0.5 
    pred = np.mean(predict(x, model, 0.5))
    pred_group = [np.mean(predict(u, model, 0.5)) for u in grouped_x]
    dd_half = 0
    for p in pred_group:
        dd_half += np.absolute(pred - p)

    # SDD
    tau = np.random.random((20,))
    pred = [np.mean(predict(x, model, t)) for t in tau]
    pred_group = []

    for x_group in grouped_x:
        pred_group.append([np.mean(predict(x_group, model, t)) for t in tau])
        
    pred_group = np.array(pred_group)

    # SDD
    sdd = 0
    for i,  _ in enumerate(grouped_x):
        sdd += np.mean(np.absolute(pred - pred_group[i, :]))

    # SPDD
    spdd = 0
    for i, _ in enumerate(grouped_x):
        for j, _ in enumerate(grouped_x):
            if i != j:
                spdd += np.mean(np.absolute(pred_group[j, :] - pred_group[i, :]))

    return dd_half, sdd, spdd
    
