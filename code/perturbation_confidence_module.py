import numpy as np
import scipy as sp

"""
Set of functions to generate random perturbation vectors
that will not change the output of the network
but will provoke a desired change of confidence
and contain random confidence-orthogonal components.
The perturbation vectors are normalized so as to be of equal norm among
all confidence change levels and all networks.
"""

def generate_random_perturbation_vector(prototype, norm, delta_conf):
    """
    Parameters
    ----------
    prototype : a perturbation prototype object
    delta_conf : the  confidence change that this perturbation will cause
    norm : the norm of the perturbation vector to be generated
    Returns
    -------
    a perturbation vector of shape [1, n_units]
    """
    # compute the confidence component coefficient to yield the desired change
    q_c = prototype["confidence_component_vector"]
    q_c_squared = np.matmul(q_c, q_c.T)[0,0]
    coef_q_c = delta_conf / q_c_squared
    # compute the random component vector
    q_r_subspace_basis = prototype["confidence_orthogonal_basis"]
    q_r_projected = generate_random_unit_vector(q_r_subspace_basis.shape[1])
    q_r = unproject(q_r_projected, q_r_subspace_basis)
    assert abs(np.linalg.norm(q_r, ord=2) - 1.) < 1e-3, "programmer error in computation"
    # compute the random component coefficient to satisfy the norm constraint
    coef_q_r = np.sqrt(norm ** 2 - delta_conf ** 2 / q_c_squared)
    # compute the perturbation vector in the hidden state space
    q_subspace_basis = prototype["basis"]
    q_projected = coef_q_c * q_c + coef_q_r * q_r
    q = unproject(q_projected, q_subspace_basis)
    assert abs(np.linalg.norm(q, ord=2) - norm) < 1e-3, "programmer error in computation"
    return q

def get_perturbation_prototypes(networks, confidence_decoders):
    """
    Parameters
    ----------
    networks : list of network models
    confidence_decoders : list of corresponding confidence decoders
    max_delta_conf : the maximum confidence change level to be generated
    
    Returns
    -------
    a list of prototype dictionary object for each network,
    containing the following entries
    - basis: the subspace basis matrix
    to which the perturbation vectors belong
    - confidence_component_vector: within this subspace,
    the vector component that will change confidence
    - confidence_orthogonal_basis: within this subspace,
    the subspace of components that have no effect on confidence
    """
    prototypes = []
    for i_network, network in enumerate(networks):
        decoder = confidence_decoders[i_network]
        w_ho = network.output_layer.weight.detach().numpy() # shape [1, n_units]
        w_hc = decoder.coef_ # shape [1, n_units]
        basis = get_orthogonal_complement_basis(w_ho)
        confidence_component_vector = project(w_hc, basis)
        confidence_orthogonal_basis = get_orthogonal_complement_basis(confidence_component_vector)
        prototype = dict(basis=basis,
                         confidence_component_vector=confidence_component_vector,
                         confidence_orthogonal_basis=confidence_orthogonal_basis)
        prototypes.append(prototype)
    return prototypes

def get_perturbation_norm(prototype, max_delta_conf, max_confidence_ratio):
    """
    Parameters
    ----------
    prototype : perturbation prototype
    max_delta_conf : the confidence change that should be provoked by the perturbation
    when the effect on confidence is at its strongest
    max_ratio : the ratio of the confidence-component to the total norm of the perturbation
    when the effect on confidence is at its strongest
    Returns
    -------
    The norm of the perturbation for yielding the desired effect when the effect on confidence
    is at its strongest
    """
    # compute the confidence component coefficient to yield a change of confidence
    # equal to max_delta_conf
    q_c = prototype["confidence_component_vector"]
    q_c_squared = np.matmul(q_c, q_c.T)[0,0]
    q_c_coef = max_delta_conf / q_c_squared
    # max_confidence_ratio = q_c_coef * sqrt(q_c_squared) / norm <=> norm = q_c_coef * sqrt(q_c_squared) / max_confidence_ratio
    norm = q_c_coef * np.sqrt(q_c_squared) / max_confidence_ratio
    return norm

def project(vector, subspace_basis):
    """
    - vector: shape [1, n]
    - subspace_basis: shape[n, k]
    returns projected vector: shape [1, k]
    """
    return np.matmul(vector, subspace_basis)

def unproject(projected_vector, subspace_basis):
    """
    - vector: shape [1, k]
    - subspace_basis: shape[n, k]
    returns un-projected vector: shape [1, n]
    """
    return np.matmul(projected_vector, subspace_basis.T)

def get_orthogonal_complement_basis(vector):
    """
    - vector: shape [1, n]
    - returns orthogonal complement subspace basis: shape [n, n - 1]
    """
    return sp.linalg.null_space(vector)

def generate_random_unit_vector(n_dim):
    """
    Returns a random unit vector of shape [1, n_dim]
    """
    # ref https://angms.science/doc/RM/randUnitVec.pdf
    # generate n_dim iid samples of standard normal distribution
    r = np.random.randn(1, n_dim)
    # normalize
    return r / np.linalg.norm(r, ord=2)

if __name__ == '__main__':
    # basic tests
    import training_data
    import decoding_data
    
    group_id = "b_gru11_20_shuffle_20-03-06-12-33-50"
    outcome_key = "io_confidence"
    predictor_key = "hidden_state"
    decoder_group_dir = "decoding_data/{:}".format(group_id)
    networks, network_ids = training_data.load_models_ids(group_id)
    confidence_decoders = decoding_data.load_decoders(decoder_group_dir, outcome_key, predictor_key, network_ids)
    
    prototypes = get_perturbation_prototypes(networks, confidence_decoders)
    max_delta_conf = 1.
    max_confidence_ratio = 0.9
    
    # Tests
    for i_network, prototype in enumerate(prototypes):
        perturbation_norm = get_perturbation_norm(prototype, max_delta_conf, max_confidence_ratio)
        for delta_conf in [0, max_delta_conf / 2., max_delta_conf]:
            for i in range(10):
                q = generate_random_perturbation_vector(prototype, perturbation_norm, delta_conf)
                # print("np.linalg.norm(q, ord=2)", np.linalg.norm(q, ord=2))
                # generated perturbation should have the calculated norm
                assert abs(np.linalg.norm(q, ord=2) - perturbation_norm) < 1e-3
                decoder = confidence_decoders[i_network]
                # generated perturbation should yield the desired confidence change
                actual_delta_conf = (decoder.predict(q) - decoder.intercept_)[0,0]
                assert abs(actual_delta_conf - delta_conf) < 1e-3
                
                
    
    
