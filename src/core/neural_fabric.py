#
# File: src/core/neural_fabric.py
#
from __future__ import annotations
import math
import random
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Optional, Any

# --- Constants ---
MAX_POWER_BUDGET: float = 1000000.0  # Increased power budget
NEURON_RESTING_COST: float = 0.01
NEURON_FIRING_COST: float = 1.0
SYNAPSE_TRANSMISSION_COST: float = 0.05
FIRING_THRESHOLD: float = 0.3  # Lowered from 1.0 to make neuron firing more likely
ACTIVATION_DECAY_RATE: float = 0.2
HEBBIAN_LEARNING_RATE: float = 0.01
SYNAPSE_WEIGHT_DECAY_RATE: float = 0.001
INITIAL_WEIGHT_RANGE: tuple[float, float] = (0.4, 0.6)
CONNECTION_PROBABILITY: float = 0.05
K_WINNERS_PERCENT: float = 0.02

class PowerBudgetExceededError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class Neuron:
    def __init__(self, neuron_id: int, layer_id: int, fabric: 'NeuralFabric'):
        self.neuron_id: int = neuron_id
        self.layer_id: int = layer_id
        self.fabric: 'NeuralFabric' = fabric
        self.activation_potential: float = 0.0
        self.firing_state: bool = False
        self.fabric.consume_energy(NEURON_RESTING_COST, "Neuron Creation")

    def update_state(self) -> None:
        if self.firing_state:
            self.fabric.consume_energy(NEURON_FIRING_COST, f"Neuron {self.layer_id}:{self.neuron_id} Fire")
        self.activation_potential *= (1.0 - ACTIVATION_DECAY_RATE)
        self.fabric.consume_energy(NEURON_RESTING_COST, f"Neuron {self.layer_id}:{self.neuron_id} Rest")

    def reset_cycle(self) -> None:
        self.firing_state = False

class Synapse:
    def __init__(self, pre_neuron: Neuron, post_neuron: Neuron, fabric: 'NeuralFabric'):
        self.pre_neuron: Neuron = pre_neuron
        self.post_neuron: Neuron = post_neuron
        self.fabric: 'NeuralFabric' = fabric
        self.weight: float = random.uniform(*INITIAL_WEIGHT_RANGE)

    def transmit(self) -> None:
        if self.pre_neuron.firing_state:
            signal_strength = self.weight
            self.post_neuron.activation_potential += signal_strength
            self.fabric.consume_energy(SYNAPSE_TRANSMISSION_COST, "Synapse Transmission")

    def update_weight(self, learning_modifier: float = 1.0) -> None:
        if self.pre_neuron.firing_state and self.post_neuron.firing_state:
            effective_learning_rate = HEBBIAN_LEARNING_RATE * learning_modifier
            self.weight += effective_learning_rate * (1.0 - self.weight)
        self.weight *= (1.0 - SYNAPSE_WEIGHT_DECAY_RATE)
        if self.weight > 1.0: self.weight = 1.0
        if self.weight < 0.0: self.weight = 0.0

class Layer:
    def __init__(self, layer_id: int, num_neurons: int, fabric: 'NeuralFabric', input_shape=None, rf_size=7):
        self.layer_id: int = layer_id
        self.fabric: 'NeuralFabric' = fabric
        self.neurons: List[Neuron] = [Neuron(i, layer_id, fabric) for i in range(num_neurons)]
        self.bottom_up_synapses: List[Synapse] = []
        self.input_shape = input_shape  # (H, W) for 2D layers
        self.rf_size = rf_size  # receptive field size
        self.sparse_weights = None  # Will be initialized for local connectivity
        if input_shape is not None:
            self._init_local_sparse_weights()

    def _init_local_sparse_weights(self):
        # Each neuron connects only to a local patch in the input
        H, W = self.input_shape
        N = len(self.neurons)
        rf = self.rf_size
        rows, cols, data = [], [], []
        for n in range(N):
            # Map neuron index to 2D position
            y = n // W
            x = n % W
            for dy in range(-rf//2, rf//2+1):
                for dx in range(-rf//2, rf//2+1):
                    iy, ix = y+dy, x+dx
                    if 0 <= iy < H and 0 <= ix < W:
                        idx = iy*W + ix
                        rows.append(n)
                        cols.append(idx)
                        data.append(np.random.uniform(*INITIAL_WEIGHT_RANGE))
        shape = (N, H*W)
        self.sparse_weights = sp.csr_matrix((data, (rows, cols)), shape=shape)

    def process_cycle(self, input_vector=None) -> None:
        num_neurons = len(self.neurons)
        if num_neurons == 0:
            return
        # Only compute for neurons with significant input (sparse/event-driven)
        if input_vector is not None and self.sparse_weights is not None:
            # Skip if all input is zero
            if np.count_nonzero(input_vector) == 0:
                return
            # Local receptive field: only compute weighted sum for local patch (sparse)
            potentials = self.sparse_weights @ input_vector
            # Winner-take-all: only top k fire
            num_winners = max(1, int(num_neurons * K_WINNERS_PERCENT))
            winner_indices = np.argpartition(potentials, -num_winners)[-num_winners:]
            # Batch update firing state and activation potential
            firing_mask = np.zeros(num_neurons, dtype=bool)
            firing_mask[winner_indices] = True
            for i, neuron in enumerate(self.neurons):
                neuron.firing_state = firing_mask[i]
                if firing_mask[i]:
                    neuron.activation_potential = potentials[i]
            for i in winner_indices:
                self.neurons[i].update_state()
            return
        # Dense case (fallback)
        potentials = np.array([n.activation_potential for n in self.neurons])
        active_indices = np.where(potentials > 1e-3)[0]
        if len(active_indices) == 0:
            return
        num_winners = max(1, int(num_neurons * K_WINNERS_PERCENT))
        eligible_indices = np.where(potentials >= FIRING_THRESHOLD)[0]
        if len(eligible_indices) > num_winners:
            eligible_potentials = potentials[eligible_indices]
            top_k_in_eligible = np.argsort(eligible_potentials)[-num_winners:]
            winner_indices = eligible_indices[top_k_in_eligible]
        else:
            winner_indices = eligible_indices
        firing_mask = np.zeros(num_neurons, dtype=bool)
        firing_mask[winner_indices] = True
        for i, neuron in enumerate(self.neurons):
            neuron.firing_state = firing_mask[i]
        for i in active_indices:
            self.neurons[i].update_state()  # Only update active neurons

    def update_synapses(self, learning_modifier: float = 1.0) -> None:
        for synapse in self.bottom_up_synapses:
            synapse.update_weight(learning_modifier)

    def reset_cycle(self) -> None:
        for neuron in self.neurons:
            neuron.reset_cycle()

class NeuralFabric:
    def __init__(self):
        self.layers: List[Layer] = []
        self._power_budget: float = MAX_POWER_BUDGET
        self.total_energy_consumed = 0.0

    def consume_energy(self, amount: float, source: str) -> None:
        self.total_energy_consumed += amount
        if self.total_energy_consumed > self._power_budget:
            raise PowerBudgetExceededError(
                f"Fatal: Power budget exceeded. Consumption limit of {self._power_budget} reached. "
                f"Last consumption of {amount} from '{source}' was the final straw."
            )

    def add_layer(self, num_neurons: int, input_shape=None, rf_size=7) -> Layer:
        layer = Layer(len(self.layers), num_neurons, self, input_shape=input_shape, rf_size=rf_size)
        self.layers.append(layer)
        return layer

    def connect_layers(self, pre_synaptic_layer: Layer, post_synaptic_layer: Layer) -> None:
        for post_neuron in post_synaptic_layer.neurons:
            for pre_neuron in pre_synaptic_layer.neurons:
                if random.random() < CONNECTION_PROBABILITY:
                    synapse = Synapse(pre_neuron, post_neuron, self)
                    post_synaptic_layer.bottom_up_synapses.append(synapse)

    # --- FIX: Re-added the missing _reset_cycles method ---
    def _reset_cycles(self) -> None:
        """Iterates through layers and resets the cycle for all neurons."""
        for layer in self.layers:
            layer.reset_cycle()
    
    def _propagate(self, from_layer_id: int) -> None:
        """Propagates signals from a specific layer to the next."""
        if from_layer_id + 1 >= len(self.layers):
            return
            
        next_layer = self.layers[from_layer_id + 1]
        # Reset potential for the upcoming accumulation
        for neuron in next_layer.neurons:
            neuron.activation_potential = 0.0

        # Transmit signals from firing neurons in the 'from_layer'
        from_layer = self.layers[from_layer_id]
        for pre_neuron in from_layer.neurons:
            if pre_neuron.firing_state:
                # This is inefficient, a better way would be to store outgoing synapses per neuron
                for synapse in next_layer.bottom_up_synapses:
                    if synapse.pre_neuron is pre_neuron:
                        synapse.transmit()

    def _apply_learning(self, learning_modifier: float = 1.0) -> None:
        for layer in self.layers[1:]:
            layer.update_synapses(learning_modifier)

    def process_bottom_up(self, sensory_input: List[float], learning_modifier: float = 1.0) -> None:
        if not self.layers:
            return
        input_layer = self.layers[0]
        if len(sensory_input) != len(input_layer.neurons):
            raise ValueError("Input length must match input layer size.")
        self._reset_cycles()
        # Only set activation for nonzero input (sparse)
        sensory_input = np.array(sensory_input)
        nonzero_indices = np.where(np.abs(sensory_input) > 1e-3)[0]
        if len(nonzero_indices) == 0:
            return
        for i in nonzero_indices:
            input_layer.neurons[i].activation_potential = sensory_input[i]
        for layer in self.layers:
            # Only process if there is significant input
            if hasattr(layer, 'sparse_weights') and layer.sparse_weights is not None:
                input_vec = np.array([n.activation_potential for n in layer.neurons])
                if np.count_nonzero(input_vec) == 0:
                    continue
                layer.process_cycle(input_vec)
            else:
                layer.process_cycle()
            self._propagate(layer.layer_id)
        self._apply_learning(learning_modifier)
