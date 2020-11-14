# Call Options:
def mc_call_geom_tf(enable_greeks=True):
    (S0, K, dt, T, sigma, r, dw, S_i) = tf_gbm_paths()
    A = tf.pow(tf.reduce_prod(S_i, axis=1), dt / T)
    payout = tf.maximum(A - K, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)
    target_calc = [npv]
    if enable_greeks:
        greeks = tf.gradients(npv, [S0, sigma, r, K, T])
        target_calc += [greeks]

    def pricer(S_zero, strk, maturity, volatility, riskfrate, seed, iterations, timesteps):
        if seed != 0:
            np.random.seed(seed)
        stdnorm_random_variates = np.random.randn(iterations, timesteps)
        with tf.Session() as sess:
            delta_t = maturity / timesteps
            res = sess.run(target_calc,
                           {
                               S: S_zero,
                               K: strk,
                               r: riskfrate,
                               sigma: volatility,
                               dt: delta_t,
                               T: maturity,
                               dw: stdnorm_random_variates
                           })
            return res

    return pricer
