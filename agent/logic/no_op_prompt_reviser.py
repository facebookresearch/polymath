from agent.logic.prompt_reviser import PromptReviser

class NoOpPromptReviser(PromptReviser):
    """
    A no-operation implementation of PromptReviser that does nothing.
    Used when no prompt reviser is needed but the interface must be fulfilled.
    """

    async def revise(self, constraints_prompt: str, code: str, error_details: str) -> str:
        return constraints_prompt

    def compare(self, solution: Dict[str, Any], outcome: Optional[str]) -> Tuple[bool, str]:
        return True, ""

    def on_failure(self) -> bool:
        return False

    def reset(self) -> None:
        pass
