# File: src/cognitive/architecture/learning_utils.py
"""
Reusable learning logic for error-driven continual learning, reconstruction diagnostics, and memory management.
"""
import numpy as np
from src import DEBUG, debug_print

RECON_ERROR_THRESHOLD = 2.0  # Raised to avoid premature pattern deletion

def process_letter_sample(
    letter,
    stroke_sdr,
    compositional_layer,
    cognitive_core=None,
    use_memory_manager=False,
    verbose=True
):
    """
    Encodes, reconstructs, and (optionally) updates memory for a letter sample.
    Returns: dict with keys: is_correct, recon_error, active_neurons, concept_sdr, recon_sdr
    """
    concept_sdr, topk = compositional_layer.encode(stroke_sdr)
    recon_sdr = compositional_layer.decode(concept_sdr)
    recon_error = np.linalg.norm(stroke_sdr - recon_sdr) / (np.linalg.norm(stroke_sdr) + 1e-8)
    active_neurons = np.where(concept_sdr > 0)[0]
    # --- LEARNING: Reinforce compositional layer weights based on error ---
    if recon_error > RECON_ERROR_THRESHOLD:
        compositional_layer.reinforce(topk, stroke_sdr, lr=0.01)
        if verbose:
            debug_print(f"[LEARNING] Hebbian update applied for '{letter}' (recon error: {recon_error:.3f}).")
    else:
        # Optionally reinforce correct patterns as well (positive reinforcement)
        compositional_layer.reinforce(topk, stroke_sdr, lr=0.005)
    # Optionally update memory manager (if provided)
    if use_memory_manager and cognitive_core is not None:
        is_correct = recon_error <= RECON_ERROR_THRESHOLD
        compressed = cognitive_core.compression.compress(stroke_sdr)
        schema = cognitive_core.schema.generate(letter)
        if not is_correct:
            cognitive_core.memory_manager.mark_incorrect(compressed, schema)
            if verbose:
                debug_print(f"[LEARNING] Pattern for '{letter}' marked for relearning (high recon error: {recon_error:.3f}).")
        else:
            cognitive_core.memory_manager.store(stroke_sdr, letter, error=0, reward=1, verified=True)
            if verbose:
                debug_print(f"[LEARNING] Pattern for '{letter}' stored as verified (recon error: {recon_error:.3f}).")
    if verbose:
        debug_print(f"Letter '{letter}': Recon error={recon_error:.3f}, Active neurons={len(active_neurons)}")
    return {
        'is_correct': recon_error <= RECON_ERROR_THRESHOLD,
        'recon_error': recon_error,
        'active_neurons': active_neurons,
        'concept_sdr': concept_sdr,
        'recon_sdr': recon_sdr
    }

