from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock
from agent.logic.prolog_engine_strategy import PrologEngineStrategy
from agent.logic.engine_strategy import SolverOutcome, SolverConstraints


class TestPrologEngineStrategy(IsolatedAsyncioTestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None
        self.logger_factory = lambda name: Mock()

    def test_extract_variables_and_num(self):
        code = """
        solve :-
            Vars = [Alice, Bob, Cat, Dog],
            Vars ins 1..2,
            Names = [Alice, Bob],
            Pets = [Cat, Dog],
            all_different(Names),
            all_different(Pets),
        """
        variables = PrologEngineStrategy._extract_variables(code)
        num = PrologEngineStrategy._extract_num(code)
        self.assertEqual(variables, ["Alice", "Bob", "Cat", "Dog"])
        self.assertEqual(num, 2)

    def test_parse_prolog_solution(self):
        ints = [1, 2, 1, 2]
        vars = ["Alice", "Bob", "Cat", "Dog"]
        n = 2
        res = PrologEngineStrategy._parse_prolog_solution(ints, vars, n)
        expected = "1 - Alice, Cat\n2 - Bob, Dog"
        self.assertEqual(res, expected)

    def test_parse_solver_output_success(self):
        strategy = PrologEngineStrategy(self.logger_factory, "puzzle text", "json")
        stdout = SolverConstraints("[1,2]", ["Alice", "Bob"], 2)
        exit_code = 0
        outcome, res = strategy.parse_solver_output(exit_code, stdout, "")
        self.assertEqual(outcome, SolverOutcome.SUCCESS)
        self.assertIsNotNone(res, "Result should not be None")
        assert res is not None
        self.assertIn("Alice", res)
        self.assertIn("Bob", res)

    def test_parse_solver_output_failure(self):
        mock_logger = Mock()
        strategy = PrologEngineStrategy(lambda _: mock_logger, "puzzle text", "json")
        stdout = SolverConstraints("", [], 0)
        exit_code = 1
        outcome, res = strategy.parse_solver_output(exit_code, stdout, "error")
        self.assertEqual(outcome, SolverOutcome.FATAL)
        self.assertIsNone(res)
        mock_logger.error.assert_called_once()


    async def test_6x6(self):

        prolog_code ="""
    :- use_module(library(clpfd)).
    solve :-
        % Extract all entities from the puzzle
        Vars = [Alice, Carol, Eric, Peter, Bob, Arnold,
            Classical, HipHop, Jazz, Pop, Rock, Country,
            Sarah, Penny, Aniya, Janelle, Kailyn, Holly,
            AliceChild, Fred, Timothy, Bella, Samantha, Meredith,
            VeryShort, Short, Tall, VeryTall, SuperTall, Average,
            Bird, Dog, Horse, Rabbit, Cat, Fish],

        Vars ins 1..6,
        Names = [Alice, Carol, Eric, Peter, Bob, Arnold],
        MusicGenres = [Classical, HipHop, Jazz, Pop, Rock, Country],
        Mothers = [Sarah, Penny, Aniya, Janelle, Kailyn, Holly],
        Children = [AliceChild, Fred, Timothy, Bella, Samantha, Meredith],
        Heights = [VeryShort, Short, Tall, VeryTall, SuperTall, Average],
        Animals = [Bird, Dog, Horse, Rabbit, Cat, Fish],

        % Add constraints for each group
        all_different(Names),
        all_different(MusicGenres),
        all_different(Mothers),
        all_different(Children),
        all_different(Heights),
        all_different(Animals),

        % Clue 1: The person who loves pop music is the cat lover.
        Pop #= Cat,
        % Clue 2: The rabbit owner is directly left of the person whose mother's name is Aniya.
        Rabbit #= Aniya - 1,
        % Clue 3: The person whose mother's name is Holly is directly left of Carol.
        Holly #= Carol - 1,
        % Clue 4: The person whose mother's name is Holly has a child named Alice.
        AliceChild #= Holly,
        % Clue 5: The person whose mother's name is Holly loves classical music.
        Classical #= Holly,
        % Clue 6: The person who loves jazz music has mother named Sarah.
        Jazz #= Sarah,
        % Clue 7: The person whose child is Meredith is somewhere to the right of the person whose mother's name is Aniya.
        Meredith #> Aniya,
        % Clue 8: The person who is super tall has mother named Holly.
        SuperTall #= Holly,
        % Clue 9: The person who is the mother of Timothy is Bob.
        Timothy #= Bob,
        % Clue 10: The person who is very short is somewhere to the left of the person whose mother's name is Aniya.
        VeryShort #< Aniya,
        % Clue 11: Eric is the fish enthusiast.
        Eric #= Fish,
        % Clue 12: The person whose child is Samantha is somewhere to the right of the person who is very tall.
        Samantha #> VeryTall,
        % Clue 13: The person who loves rock music has mother named Janelle.
        Rock #= Janelle,
        % Clue 14: There is one house between the person who keeps horses and the person whose child is Meredith.
        abs(Horse - Meredith) #= 2,
        % Clue 15: The person whose child is Bella is somewhere to the right of Peter.
        Bella #> Peter,
        % Clue 16: The fish enthusiast is somewhere to the left of the bird keeper.
        Fish #< Bird,
        % Clue 17: The fish enthusiast is somewhere to the right of the person whose child is Alice.
        Fish #> AliceChild,
        % Clue 18: There is one house between the person whose child is Bella and the person who loves rock music.
        abs(Bella - Rock) #= 2,
        % Clue 19: The person who is short is the cat lover.
        Short #= Cat,
        % Clue 20: Alice is directly left of the person who loves classical music.
        Alice #= Classical - 1,
        % Clue 21: The person whose child is Bella has mother named Aniya.
        Bella #= Aniya,
        % Clue 22: There are two houses between the person whose mother is Penny and the person who is short.
        abs(Penny - Short) #= 3,
        % Clue 23: The person who loves hip-hop music is in the first house.
        HipHop #= 1,
        % Clue 24: Carol is the person who is tall.
        Carol #= Tall,

        labeling([], Vars),
        write(Vars), nl.
        """

        # Create the strategy instance
        strategy = PrologEngineStrategy(self.logger_factory, "6x6 puzzle", "json")

        # Generate solver constraints
        solver_spec = await strategy.generate_solver_constraints(prolog_code)
        self.assertIsNotNone(solver_spec, "Solver spec should not be None")
        assert solver_spec is not None

        expected_vars = [1, 3, 5, 2, 4, 6, #Names (Alice -> house 1, Carol -> house 3, Eric -> house 5, ...)
                         2, 1, 6, 4, 5, 3, #MusicGenres
                         6, 1, 3, 5, 4, 2, #Mothers
                         2, 1, 4, 3, 6, 5, #Children
                         1, 4, 3, 5, 2, 6, #Heights
                         6, 1, 3, 2, 4, 5  #Animals
                         ]

        exit_code = 0
        stdout = SolverConstraints(
            str(expected_vars),
            solver_spec.variables,
            solver_spec.nb_categories
        )
        outcome, res = strategy.parse_solver_output(exit_code, stdout, "")
        self.assertEqual(outcome, SolverOutcome.SUCCESS)

        expected_res = "1 - Alice, HipHop, Penny, Fred, VeryShort, Dog\n2 - Peter, Classical, Holly, AliceChild, SuperTall, Rabbit\n3 - Carol, Country, Aniya, Bella, Tall, Horse\n4 - Bob, Pop, Kailyn, Timothy, Short, Cat\n5 - Eric, Rock, Janelle, Meredith, VeryTall, Fish\n6 - Arnold, Jazz, Sarah, Samantha, Average, Bird"
        self.assertEqual(res, expected_res)
