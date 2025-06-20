"""
CognitiveCore: Central controller for AGI modules, following the law of least action (energy efficiency).
Activates modules only as needed, routes information, and manages memory, reasoning, and learning.
"""
from .probabilistic_reasoner import ProbabilisticReasoner
from .bayesian_inference import BayesianInferenceEngine
from .deductive_reasoner import DeductiveReasoner
from .inductive_learner import InductiveLearner
from .causal_inference import CausalInferenceEngine
from .counterfactual_simulator import CounterfactualSimulator
from .predictive_coding import PredictiveCodingSystem
from .hebbian_learning import HebbianLearningModule
from .reinforcement_learner import ReinforcementLearner
from .pattern_compression import PatternCompressionEngine
from .memory_replay import MemoryReplayLoop
from .language_learner import LanguageLearner
from .schema_generator import SchemaGenerator
from .internal_language import InternalLanguageGenerator
from .memory_manager import MemoryManager

class CognitiveCore:
    def __init__(self):
        self.prob_reasoner = ProbabilisticReasoner()
        self.bayes_engine = BayesianInferenceEngine()
        self.deductive = DeductiveReasoner()
        self.inductive = InductiveLearner()
        self.causal = CausalInferenceEngine()
        self.counterfactual = CounterfactualSimulator()
        self.predictive = PredictiveCodingSystem()
        self.hebbian = HebbianLearningModule()
        self.rl = ReinforcementLearner()
        self.compression = PatternCompressionEngine()
        self.replay = MemoryReplayLoop()
        self.language = LanguageLearner()
        self.schema = SchemaGenerator()
        self.internal_lang = InternalLanguageGenerator()
        self.memory_manager = MemoryManager(self.compression, self.schema)

    def process(self, sensory_input, context=None):
        # Example: only activate modules as needed (event-driven, not all at once)
        # This is a stub for the main cognitive loop
        # 1. Predictive coding: fast, low-power, always-on
        prediction, error = self.predictive.predict(sensory_input)
        if error > self.predictive.threshold:
            # 2. If error is high, engage higher-level modules
            beliefs = self.bayes_engine.infer(sensory_input, context)
            deduction = self.deductive.reason(beliefs)
            induction = self.inductive.learn(sensory_input)
            # ...and so on, only as needed
        # 3. Memory replay and compression can run during idle cycles
        self.replay.replay()
        self.compression.compress(sensory_input)
        # 4. Language and schema modules for abstraction and communication
        schema = self.schema.generate(context)
        internal_narrative = self.internal_lang.generate(schema)
        # 5. Store only salient, compressed, schema-level memories
        self.memory_manager.store(sensory_input, context)
        # 6. Return minimal action/energy path output
        return prediction, error, beliefs, deduction, induction, schema, internal_narrative
