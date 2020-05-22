import numpy as np
import wasserstein_fairness.basic_costs as costs
def metrics(x, grouped_data, alpha, beta, lr, batch_size):

    # Retrive coeff vector
    filename = f'weights/alpha-{alpha}-beta-{beta}-lr-{lr}-batch_size-{batch_size}.npy'
    theta = np.load(filename)
    
    # DD-0.5 
    pred = np.mean(costs.predict(x, theta, 0.5))
    pred_group = [np.mean(costs.predict(u, theta, 0.5)) for u in grouped_data]
    dd_half = 0
    for p in pred_group:
        dd_half += np.absolute(pred - p)

    # SDD
    tau = np.random.random((20,))
    pred = [np.mean(costs.predict(x, theta, t)) for t in tau]
    pred_group = []

    for x_group in grouped_data:
        pred_group.append([np.mean(costs.predict(x_group, theta, t)) for t in tau])
        
    pred_group = np.array(pred_group)

    # SDD
    sdd = 0
    for i,  _ in enumerate(grouped_data):
        sdd += np.mean(np.absolute(pred - pred_group[i, :]))

    # SPDD
    spdd = 0
    for i, _ in enumerate(grouped_data):
        for j, _ in enumerate(grouped_data):
            if i != j:
                spdd += np.mean(np.absolute(pred_group[j, :] - pred_group[i, :]))

    return dd_half, sdd, spdd


