# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

from agent.logic.engine_strategy import EngineStrategy, SolverOutcome

from agent.logic.logic_py_smt_harness_generator import LogicPySMTHarnessGenerator

from agent.symex.module_with_type_info_factory import ModuleWithTypeInfoFactory
from agent.symex.propagate_unique import PropagateUnique
from libcst import MetadataWrapper, Module


# Prefix containing Logic.py helper functions that do not exist in standard
# Python. These are included to provide information for type checkers, their
# semantics are explicitly modelled in symbolic exeuction.
_PYTHON_CODE_PREFIX: str = """E = typing.TypeVar("E")
def some(elements: list[E]) -> E:
    return elements[0]

T = typing.TypeVar("T")
class Unique(typing.Generic[T]):
    pass

"""

# Message sent to the model to generate the second (conclusion) constraint.
_CONCLUSION_MESSAGE: str = """Now create the `conclusion` function in the same manner for the conclusion in the logic question:

```
def conclusion(universe: Universe) -> None:
    ...
```

Please remember the following instructions:
- NEVER use comprehensions to express your constraints.
- NEVER use `isinstance(...)`, this function is not available.
- All variables are read-only, never create temporary variables, only assertions on attributes.
- Only use loops over lists, i.e. do not use e.g. `for _ in range(2)`. If you need to repeat something multiple times, just repeat the statements and create multiple variables.
"""

# Message sent to the model to generate the basic data structure over which
# premise and conclusion are expressed.
_DATA_STRUCTURE_MESSAGE: str = """Now begin! Here is the logic question:

```
Premise:
{premise}

Conclusion:
{conclusion}
```
"""

# Message sent to the model to generate the first (premis) constraint.
_PREMISE_MESSAGE: str = """Now you must create a premise function which checks that the premise described in the text holds for a given universe. The function will have the following signature:

```
def premise(universe: Universe) -> None:
    ...
```

I'll show you the kind of constraints you can express in the following examples:


"If Jane is not either a ballerina or she owns a dog, then..."
```
jane = some(universe.owner)
assert jane.name == "Jane"
dog = some(dog)
if not (jane.is_ballerina or dog.owner == jane):
    assert ...
```

"Children either have a parent who likes dancing, or they are not older than 10."
```
for child in universe.children:
    parent = some(universe.parents)
    assert (child in parent.children and parent.likes_dancing) or child.age <= 10
```

"Pets can have long hair. Pets also may be shy."
```
pet_with_long_hair = some(universe.pets)
assert pet_with_long_hair.has_long_hair

shy_pet = some(universe.pets)
assert shy_pet.is_shy
```

"If Sam is a coach or works at other schools or both, then Sam does not either have lunch at school or drives to work."
```
sam = some(universe.person)
assert sam.name == "Sam"
if sam.is_coach or_both sam.works_at_other_schools:
    assert not (sam.has_lunch_at_school or sam.drives_to_work)
```

"If an applicant has a masters' degree, and once he has filled out the necessary forms, he may have a chance to be shortlisted."
```
for applicant in universe.applicants:
    if not applicant.has_masters_degree or not applicant.has_filled_forms:
        assert not applicant.has_chance_to_be_shortlisted
```


Elements always have a value and will never be `None`. If you want to just assert that an element of some type exists, simply use:

```
# Any teams exist:
assert len(universe.teams) > 0

# A team with a given attribute exists:
the_team = some(universe.teams)
assert the_team.is_sponsored
```

Please also adhere to the following instructions:
- NEVER use comprehensions to express your constraints.
- NEVER use `isinstance(...)`, this function is not available.
- All variables are read-only, never create temporary variables, only assertions on attributes.
- Only use loops over lists, i.e. do not use e.g. `for _ in range(2)`. If you need to repeat something multiple times, just repeat the statements and create multiple variables.

Now generate the constraints to assert that the given universe adheres to the constraints in the logic question.
"""

# Sent if the premise was self-contradictory.
_RETRY_MESSAGE: str = (
    "The premise function you generated was self-contradictory. Please review it and try to spot the error. We will go through the steps of generating the premise and conclusion functions from scratch again."
)

# Basic system prompt explaining Logic.py and majority of data structure
# generation prompt.
_SYSTEM_PROMPT: str = """You are an expert logic agent with access to a propositional logic solver tool that has a convenient interface optimised for you. Your boss has given you a logic question that he thinks can be mapped to your logic solver tool. You will answer this question in two steps:

1. Define the data type to represent the universe of the question
2. Define the logical constraints over this universe based on the question’s text

I will walk you through these steps one by one. Do not attempt to answer the question or proceed to a following step before I tell you to do so.

Your solver tool has a Python API that allows you to define custom data types. Use these data types to express properties of objects and relations between objects. I'll teach you how to use this tool using the following examples: You can also mark properties as `Unique` in order to refer to the same objects in your constraints. As an example, assume you have the following text in a logic question: "Peter is a manager in the data science team. He cannot program and has a performance score of 17.25. He is 34 years old."

This could be mapped to the following example data structure:

```
# Data structure defined in first step
class Person:
    name: Unique[str]
    age: int
    can_program: bool
    performance_score: float

class Team:
    name: Unique[str]
    manager: Person

class Universe:
    people: list[Person]
    teams: list[Team]
```

Create as few attributes as possible to express the constraint. NEVER do something like this:

```
class Employee:
    works_from_home: bool
    has_a_meeting: bool
    appears_in_office: bool # NEVER DO THIS! `appears_in_office` is just a negation of `works_from_home`!
```

Keep your data structures simple. Do not use a new class or use inheritance if a `bool` attribute is sufficient:

```
# DO NOT DO THIS:
class Animal:
    is_fast: bool

class Dog(Animal):
    pass

class Cat(Animal):
    pass


# Do this instead:
class Animal:
    is_fast: bool
    is_dog: bool
    is_cat: bool
```

Now specify the type of the universe using this syntax. Please also adhere to the following instructions:

- When asked to generate Python code, only ever provide a single code snippet surrounded by a ``` block, so that I can easily parse it.
- Define only the Python classes for now, no variables or constraints.
- Add a `list` of every custom type you declare to the `Universe` class.
"""

# Namespace for Z3 expected output results.
z3_output_patterns = SimpleNamespace()
# Incorrect conclusion output.
z3_output_patterns._FALSE = """sat
unsat
sat
"""
# Self-contradictory premise output.
z3_output_patterns._RETRY = """unsat
unsat
unsat
"""
# Correct conclusion output.
z3_output_patterns._TRUE = """sat
sat
unsat
"""
# Independent conclusion output.
z3_output_patterns._UNKNOWN = """sat
sat
sat
"""


class Z3ConclusionCheckEngineStrategy(EngineStrategy):
    """
    Given a premise and a conclusion, checks in Z3 whether:
    1) Conclusion follows from the premise.
    2) Premise contradicts conclusion.
    3) Premise is self-contradictory.
    4) Premise and conclusion are independent.
    """

    def __init__(
        self, logger_factory: Callable[[str], Logger], premise: str, conclusion: str
    ) -> None:
        """
        Args:
            logger_factory (Callable[[str], Logger]): Log configuration.
            premise (str): Natural language premise.
            conclusion (str): Natural language conclusion.
        """
        self.__logger: Logger = logger_factory(__name__)
        self.__premise: str = premise
        self.__conclusion: str = conclusion

    @property
    def constraints_prompt(self) -> list[str]:
        return [_PREMISE_MESSAGE, _CONCLUSION_MESSAGE]

    @property
    def data_structure_prompt(self) -> str:
        return _DATA_STRUCTURE_MESSAGE.format(
            premise=self.__premise, conclusion=self.__conclusion
        )

    async def generate_solver_constraints(
        self, module: Module, metadata: Optional[MetadataWrapper]
    ) -> str:
        if metadata is None:
            raise ValueError("SMT back-end needs type information enabled")

        preprocessed: Module = await PropagateUnique.preprocess(metadata)
        wrapper: MetadataWrapper = await ModuleWithTypeInfoFactory.create_module(
            preprocessed.code
        )
        return LogicPySMTHarnessGenerator.generate(wrapper)

    def generate_solver_invocation_command(self, solver_input_file: str) -> list[str]:
        return ["z3", solver_input_file]

    def get_format_prompt(self, solution: str) -> Optional[str]:
        return None

    def parse_solver_output(
        self, exit_code: int, stdout: str, stderr: str
    ) -> Tuple[SolverOutcome, Optional[str]]:
        if exit_code == 0:
            output: str
            match stdout:
                case z3_output_patterns._FALSE:
                    output = "False"
                case z3_output_patterns._RETRY:
                    return SolverOutcome.RETRY, None
                case z3_output_patterns._TRUE:
                    output = "True"
                case z3_output_patterns._UNKNOWN:
                    output = "Uncertain"
                case _:
                    raise ValueError(
                        f"Unexpected Z3 output. Stdout: `{stdout}`, Stderr: `{stderr}`"
                    )

            return SolverOutcome.SUCCESS, output

        self.__logger.warning(f"Encountered Z3 error: {stdout}\n{stderr}")
        return SolverOutcome.FATAL, None

    @property
    def python_code_prefix(self) -> str:
        return _PYTHON_CODE_PREFIX

    @property
    def retry_prompt(self) -> str:
        return _RETRY_MESSAGE

    @property
    def solver_input_file_suffix(self) -> str:
        return ".smt"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
