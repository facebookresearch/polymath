from agent.logic.prompt_reviser import PromptReviser
from agent.logic.engine_strategy import EngineStrategy

from logging import Logger
from inference.client import InferenceClient
from inference.chat_completion import Message, Role

from agent.logic.analysis.solution_comparator import SolutionComparator
from typing import Callable, Optional, Tuple
import os

class AdaptivePromptReviser(PromptReviser):
    """
    A concrete implementation of PromptReviser that adaptively modifies the constraint
    prompt based on execution errors or incorrect results.

    This reviser interacts with the LLM to regenerate improved constraint prompts
    and automatically adjusts the LLM temperature if repeated revisions fail.
    It also uses a solution comparator to evaluate prediction correctness.

    Attributes:
        engine_strategy (EngineStrategy): Strategy to generate prompts.
        client (InferenceClient): Client for interacting with the LLM.
        comparator (SolutionComparator): Comparator for evaluating solution correctness.
        max_num_revise (int): Maximum number of prompt revisions before increasing temperature.
        max_temperature (float): Maximum temperature value to use for sampling.
        temperature_step (float): Step size to increase temperature.
    """
    def __init__(
        self,
        engine_strategy: EngineStrategy,
        client: InferenceClient,
        logger_factory: Callable[[str], Logger],
        comparator: SolutionComparator,
        expected_solution: str,
        max_num_revise: int = 3,
        max_temperature: float = 0.9,
        temperature_step: float = 0.3,
    ):
        self.engine_strategy = engine_strategy
        self.client = client
        self.logger: Logger = logger_factory(__name__)
        self.comparator = comparator
        self.expected_solution = expected_solution
        self.enter_revise = False
        self.max_num_revise = max_num_revise
        self.max_temperature = max_temperature
        self.temperature_step = temperature_step
        self.num_revise = 0
        self.curr_temperature = 0.0

    async def check_and_revise(self, outcome: str, code: str):

        success, err = self.compare(outcome)

        if not success:
            self.__enter_revise = True
            await self.revise(
                os.linesep.join(self.engine_strategy.constraints_prompt),
                code,
                err
            )
            return True

        if self.enter_revise:
            self.engine_strategy.set_initial_constraints_prompt()

        return False


    async def revise(self, constraints_prompt: str, code: str, error_details: str):
        self.logger.warning("Retrying prompt revising due to incorrect result.")
        prompt = self.engine_strategy.get_revise_prompt(constraints_prompt, code, error_details)
        message = Message(Role.USER, prompt)
        _, ai_response = await self.client.create(message)
        self.engine_strategy.set_constraints_prompt(ai_response)

    def compare(self, outcome: Optional[str]) -> Tuple[bool, str] :
        success, _, err = self.comparator.compare(outcome, self.expected_solution)
        return success, err


    def on_failure(self) -> bool:

        self.client.conversation.clear()
        self.num_revise += 1
        if self.num_revise >= self.max_num_revise:
            if self.curr_temperature < self.max_temperature :
                self.curr_temperature += self.temperature_step
                self.client.set_temperature(self.curr_temperature)
                self.num_revise = 0
                self.engine_strategy.reset_constraints_prompt()
            else:
                return False

        return True



    def reset(self) -> None:
        self.num_revise = 0
        self.curr_temperature = 0.0
        self.client.reset_temperature()
