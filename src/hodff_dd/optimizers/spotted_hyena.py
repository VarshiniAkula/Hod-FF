"""
A tiny, practical SHO variant that *initializes* the final Dense layer weights.
We treat each candidate as a small perturbation around the current weights and
pick the best using a simplified hyena position update. This is lightweight and
sufficient for small-data demos.
"""
import numpy as np
import tensorflow as tf

def get_last_dense_td(model):
    # Returns the TimeDistributed Dense(2, softmax) layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            if isinstance(layer.layer, tf.keras.layers.Dense) and layer.layer.units == 2:
                return layer
    raise RuntimeError("Could not find TimeDistributed(Dense(2)) head.")

def _evaluate_loss(model, dataset, steps=2):
    # quick evaluation on a few batches
    loss = tf.keras.metrics.Mean()
    for i, (x, y) in enumerate(dataset):
        if i >= steps: break
        preds = model(x, training=False)
        l = tf.keras.losses.categorical_crossentropy(y, preds)
        loss.update_state(tf.reduce_mean(l))
    return float(loss.result().numpy())

def sho_initialize_final_head(model, dataset, population=6, iterations=5, verbose=True):
    head = get_last_dense_td(model)
    # Current weights
    W, b = head.layer.get_weights()  # shapes: (in_dim, 2), (2,)
    in_dim = W.shape[0]

    # Build population around current weights
    def sample_candidate():
        return (W + 0.01*np.random.randn(in_dim, 2),
                b + 0.01*np.random.randn(2))

    pop = [sample_candidate() for _ in range(population)]

    # Helper to assign and eval
    def assign_and_eval(candidate):
        head.layer.set_weights(candidate)
        return _evaluate_loss(model, dataset, steps=2)

    # Evaluate initial fitness (lower is better)
    fitness = np.array([assign_and_eval(c) for c in pop], dtype=np.float32)

    for it in range(iterations):
        # Best index
        best_idx = int(np.argmin(fitness))
        best = pop[best_idx]
        # Update others towards best (simplified encircling)
        new_pop = []
        for i, (Wi, bi) in enumerate(pop):
            if i == best_idx:
                new_pop.append((Wi, bi))
                continue
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2*r1 - 1.0      # [-1,1]
            C = 2*r2            # [0,2]
            # encircling prey (best)
            Wb, bb = best
            D_w = np.abs(C*Wb - Wi)
            D_b = np.abs(C*bb - bi)
            new_W = Wb - A*D_w
            new_b = bb - A*D_b
            # small random exploration
            new_W += 0.005*np.random.randn(*new_W.shape)
            new_b += 0.005*np.random.randn(*new_b.shape)
            new_pop.append((new_W, new_b))
        pop = new_pop
        fitness = np.array([assign_and_eval(c) for c in pop], dtype=np.float32)
        if verbose:
            print(f"[SHO] iter {it+1}/{iterations}, best loss: {fitness.min():.4f}")

    # Set best weights
    best_idx = int(np.argmin(fitness))
    best = pop[best_idx]
    head.layer.set_weights(best)
    if verbose:
        print(f"[SHO] Done. Best loss after SHO init: {fitness.min():.4f}")
