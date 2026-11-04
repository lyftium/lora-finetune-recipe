from simulate import N_TRIALS, main


def test_token_weighted_beats_naive_mean_of_means():
    summary = main()
    assert summary["n_trials"] == N_TRIALS
    assert summary["max_abs_err_token_weighted"] < 1e-9
    assert summary["mean_abs_err_naive_mean_of_means"] > summary["mean_abs_err_token_weighted"]
