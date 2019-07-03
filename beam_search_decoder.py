import numpy as np


def beam_search(probabilities_matrix, beam_width):
    """
    Performs beam search to find the best possible sequence of words given probabilities_matrx
    :param probabilities_matrix: Probabilities matrix of shape (max_seq_len, vocab_size)
    :param beam_width: beam_width to search for
    """

    # Get required constants basis shape of probabilities matrix
    max_seq_len = probabilities_matrix.shape[0]
    vocab_size = probabilities_matrix.shape[1]

    # Get word indices and probs of occurrence of those words at time step t0
    t0_probs = probabilities_matrix[0, :]
    word_indices_t0_sorted = np.argsort(t0_probs)[::-1]
    word_probs_t0_sorted = np.sort(t0_probs)[::-1]

    # The top-k (beam-width) 1-gram sequence probabilities
    beam_sequences_probs = word_probs_t0_sorted[:beam_width].reshape(1, beam_width)
    candidate_sequences = [[word_index] for word_index in word_indices_t0_sorted[:beam_width]]

    # Take log of beam sequence probs
    log_beam_sequences_probs = np.log(beam_sequences_probs)

    # Apply beam search for remaining time steps
    for step_ind in range(1, max_seq_len):

        print ("This is step {} bro".format(step_ind))

        # Prepare log of prob_vector at t'th time step for addition with log_beam_sequence_probs
        prob_vector = probabilities_matrix[step_ind, :].reshape(1, vocab_size)
        prob_vector = np.repeat(prob_vector, beam_width, axis=0)  # O/p mat: (beam_width x vocab_size)
        log_prob_vector = np.log(prob_vector)
        print ("Shape of log_prob_vector", log_prob_vector.shape)

        # Reshape log_beam_sequences_probs for addition with log_prob_vector
        log_beam_sequences_probs = np.repeat(log_beam_sequences_probs, vocab_size, axis=0)  # O/p (vcb_size x bm_width)
        log_beam_sequences_probs = np.transpose(log_beam_sequences_probs)  # O/p mat: (beam_width x vocab_size)
        print ("Shape of log_beam_sequence_prob", log_beam_sequences_probs.shape)

        # Add log_beam_sequences_probs and log_prob_vector
        next_step_log_probs = log_beam_sequences_probs + log_prob_vector

        # Find new beam sequences
        next_step_indices_sorted = np.argsort(next_step_log_probs, axis=None)[::-1]
        next_step_log_probs_sorted = np.sort(next_step_log_probs, axis=None)[::-1]

        log_beam_sequences_probs = next_step_log_probs_sorted[:beam_width].reshape(1, beam_width)

        top_beam_indices_flattened = next_step_indices_sorted[:beam_width]
        top_beam_indices = np.array([
            [int(flattened_ind / vocab_size), flattened_ind % vocab_size]
            for flattened_ind in top_beam_indices_flattened
        ])
        print ("Shape of top_beam_indices", top_beam_indices.shape)

        values_to_fetch_from_prev_step = top_beam_indices[:, 0]
        values_to_add_in_curr_step = top_beam_indices[:, 1]
        new_candidate_sequences = []

        for prev_candidate_seq_ind, new_word_ind in zip(values_to_fetch_from_prev_step, values_to_add_in_curr_step):
            candidate_seq = candidate_sequences[prev_candidate_seq_ind] + [new_word_ind]
            new_candidate_sequences.append(candidate_seq)

        candidate_sequences = new_candidate_sequences.copy()
        print ("Elements in candidate sequences after step completion", candidate_sequences)

    max_likely_seq_ind = np.argmax(log_beam_sequences_probs)
    max_likely_sequence = candidate_sequences[max_likely_seq_ind]

    print ("Final log beam sequence probs", log_beam_sequences_probs)
    print ("Final Candidate sequences", candidate_sequences)
    print ("Max likely sequence", max_likely_sequence)

    return max_likely_sequence


if __name__ == '__main__':

    vocab_size = 7
    seq_len = 5
    beam_w = 3

    prob_matrix = np.random.rand(seq_len, vocab_size)
    print (prob_matrix.shape)

    beam_search(prob_matrix, beam_w)
