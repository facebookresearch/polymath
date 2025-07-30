from agent.logic.base_error_handler import BaseErrorHandler

class NullErrorHandler(BaseErrorHandler):
    async def revise(self, code: str, error_details: str) -> bool:
        # No-op: do not retry
        return False

    def on_failure(self) -> bool:
        return False

    def reset(self) -> None:
        pass
