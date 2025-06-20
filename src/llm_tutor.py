class LLMTutor:
    """Simulated lightweight language model for letter guidance."""
    def __init__(self):
        self.alphabet = {ch: f"This is the letter {ch.upper()}, it sounds like '{ch.lower()}'." for ch in (list('abcdefghijklmnopqrstuvwxyz'))}

    def describe_letter(self, ch: str) -> str:
        """Return a simple explanatory sentence for a letter."""
        return self.alphabet.get(ch.lower(), "Unknown letter")


