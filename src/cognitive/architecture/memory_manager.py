from src import DEBUG, debug_print

class MemoryManager:
    """
    Stores only compressed, salient, or schema-level memories. Prunes low-value or redundant data.
    Mimics human memory: lossy, selective, and abstract.
    """
    def __init__(self, compression_engine, schema_generator, max_memories=1000):
        self.compression = compression_engine
        self.schema = schema_generator
        self.max_memories = max_memories
        self.memory = []  # List of (compressed, schema, salience, is_core) tuples
        self.core_concepts = set()  # Set of hashes for core/verified concepts

    def store(self, data, context=None, error=0, reward=0, verified=False):
        debug_print("[DEBUG] Entered store() method")
        compressed = self.compression.compress(data)
        schema = self.schema.generate(context)
        salience = self.compute_salience(data, error, reward, verified)
        is_core = verified
        mem_hash = self._memory_hash(compressed, schema)
        debug_print(f"[DEBUG] Attempting to store memory: hash={mem_hash}, is_core={is_core}, salience={salience}, context={context}")
        if is_core:
            self.core_concepts.add(mem_hash)
        if self.is_salient(salience, is_core):
            self.memory.append((compressed, schema, salience, is_core, mem_hash))
            if len(self.memory) > self.max_memories:
                self.prune()
            debug_print(f"[MEMORY] Stored: {len(self.memory)} memories (core: {len(self.core_concepts)})")
        else:
            debug_print(f"[MEMORY] Not stored: salience too low or not core (salience={salience}, is_core={is_core})")

    def compute_salience(self, data, error, reward, verified):
        # High error, high reward, or verified = high salience
        return float(error) + float(reward) + (10.0 if verified else 0.0)

    def is_salient(self, salience, is_core):
        # Always store core/verified, otherwise threshold
        return is_core or salience > 0.5

    def prune(self):
        # Never prune core/verified concepts
        non_core = [m for m in self.memory if not m[3]]
        core = [m for m in self.memory if m[3]]
        # Prune lowest-salience non-core memories first
        non_core.sort(key=lambda m: m[2])
        keep = core + non_core[-(self.max_memories - len(core)):]
        self.memory = keep[-self.max_memories:]
        debug_print(f"[MEMORY] Pruned: {len(self.memory)} memories remain (core: {len(self.core_concepts)})")

    def recall(self, query=None):
        # Return best-matching memory (compressed, schema)
        if not self.memory:
            return None
        # Placeholder: return most recent
        return self.memory[-1]

    def _memory_hash(self, compressed, schema):
        # Simple hash for deduplication/core tracking
        return hash(str(compressed) + str(schema))

    def mark_incorrect(self, compressed, schema):
        # Remove or demote incorrect pattern from memory/core
        mem_hash = self._memory_hash(compressed, schema)
        self.core_concepts.discard(mem_hash)
        before = len(self.memory)
        self.memory = [m for m in self.memory if m[4] != mem_hash]
        after = len(self.memory)
        debug_print(f"[MEMORY] Marked incorrect: removed {before - after} memory (core: {len(self.core_concepts)})")

