import tensorflow as tf

def metrics(data, groups, seed, lr, wlr, epsilon, w_reg, l2_reg):
    x, y, group = data
    filename = f'saved-models/seed-{seed}-lr-{lr}-wlr-{wlr}\
            -epsilon-{epsilon}-w_reg-{w_reg}-l2_reg-{l2_reg}'
