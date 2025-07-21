from io import StringIO
from json import loads
from typing import Any


def parser(input: str) -> str:
    """
    Parses the input string and returns a formatted string.

    Args:
        input (str): The input string to be parsed.
    
    Returns:
        str: The formatted output string.
    """
    try:
        cbmc_json_output: Any = loads(input)
    except ValueError as e:
        return f"Error parsing input: {e}"
    
    parsed_output: str  = __parse_cbmc_solution(cbmc_json_output)
    return parsed_output

def __parse_cbmc_solution(cbmc_output: Any) -> str:
    """
    Extracts the solution counterexample from a CBMC JSON trace, formatted
    as a Pyton-ish expression to make it easier for the model to interpret.

    Args:
        cbmc_output (Any): CBMC JSON output.
    Returns:
        Python-ish expression equivalent to solution output struct in CBMC
        trace.
    """
    output_step: Any = [
        step
        for message in cbmc_output
        if "result" in message
        for result in message["result"]
        if result["status"] == "FAILURE"
        for step in result["trace"]
        if step["stepType"] == "output"
    ][0]
    value: Any = output_step["values"][0]

    string_builder = StringIO()
    __cbmc_value_to_string(string_builder, value, "")
    return string_builder.getvalue()

def __cbmc_value_to_string(
        string_builder: StringIO, cbmc_json_value: Any, indent: str
    ) -> None:
    """
    Formats a CBMC C output expression as a Python-ish expression, such
    that the model can easily interpret the answer by the solver.

    Args:
        string_builder (StringIO): Output to which we write the result
        expression.
        cbmc_json_value (Any): CBMC JSON output to format.
        indent (str): Current indentation prefix. Used when recursively
        invoking this method for nested members.
    """
    next_indent: str = indent + "  "
    if "members" in cbmc_json_value:
        string_builder.write(f"{indent}{{\n")
        is_first = True
        for member in cbmc_json_value["members"]:
            name: str = member["name"]
            if name.startswith("$pad"):
                continue

            if not is_first:
                string_builder.write(f",\n")
            is_first = False

            string_builder.write(f"{next_indent}{name}: ")
            __cbmc_value_to_string(
                string_builder, member["value"], next_indent
            )
        string_builder.write(f"\n{indent}}}")
    elif "elements" in cbmc_json_value:
        string_builder.write(f"[\n")
        is_first = True
        for element in cbmc_json_value["elements"]:
            if not is_first:
                string_builder.write(f",\n")
            is_first = False

            __cbmc_value_to_string(
                string_builder, element["value"], next_indent
            )
        string_builder.write(f"\n{indent}]")
    elif "data" in cbmc_json_value:
        value: str = cbmc_json_value["data"]
        if cbmc_json_value["type"] == "const char *":
            string_builder.write(f'"{value}"')
        else:
            string_builder.write(value)
if __name__ == "__main__":

    file_path = "agent/output.txt"
    with open(file_path, "r") as file:
        input_data = file.read()
    output = parser(input_data)
    print(output)