import numpy as np

def get_changes_in_prediction_by_coupled(df):
    pre_rep_pgen = df["pre_rep_pgen"].iloc[0]
    pre_other_pgen_values = df["pre_other_pgen"].unique()
    rep_lengths = df["rep_length"].unique()
    n_rep_lengths = len(rep_lengths)
    # get number of models per condition
    pre_other_pgen = pre_other_pgen_values[0]
    is_coupled = False
    n_models_independent = len(df.query("(rep_item == 1) &\
              (pre_other_pgen == @pre_other_pgen) &\
              (is_coupled == @is_coupled) &\
              (rep_length == 0)")["p_after"].to_numpy())
    is_coupled = True
    n_models_coupled = len(df.query("(rep_item == 1) &\
              (pre_other_pgen == @pre_other_pgen) &\
              (is_coupled == @is_coupled) &\
              (rep_length == 0)")["p_after"].to_numpy())
    # compute absolute change of prediction | after - before |
    shape_independent = (4, n_rep_lengths, n_models_independent)
    shape_coupled = (4, n_rep_lengths, n_models_coupled)
    p_diff_by_coupled = {
        False: np.empty(shape_independent),
        True: np.empty(shape_coupled),
    }
    for i_pre_other_pgen, pre_other_pgen in enumerate(pre_other_pgen_values):
        for i_rep_item, rep_item in enumerate([0, 1]):
            i_quadrant = i_pre_other_pgen * 2 + i_rep_item
            for is_coupled in [False, True]:
                n_models = n_models_coupled if is_coupled else n_models_independent
                p_after_s = np.empty((n_rep_lengths, n_models))
                p_before_s = np.empty((n_rep_lengths, n_models))
                for i_rep_length, rep_length in enumerate(rep_lengths):
                    df_condition = df.query("(rep_item == @rep_item) &\
                        (pre_other_pgen == @pre_other_pgen) &\
                        (is_coupled == @is_coupled) &\
                        (rep_length == @rep_length)")
                    p_after_s[i_rep_length] = df_condition["p_after"].to_numpy()
                    p_before_s[i_rep_length] = df_condition["p_before"].to_numpy()
                p_diff_by_coupled[is_coupled][i_quadrant] = \
                        abs(p_after_s - p_before_s)

    # average over 4 quadrants
    for is_coupled in [True, False]:
        p_diff_by_coupled[is_coupled] = p_diff_by_coupled[is_coupled].mean(axis=0)

    return p_diff_by_coupled
