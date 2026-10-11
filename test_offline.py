"""
Offline test suite for Data Analysis Agent.

Runs without network access or real API keys.
Validates:
1. inspect_data and run_python tools
2. AST safety guard (blocked modules, builtins, dunder attributes)
3. Matplotlib plot auto-saving and outputs directory
4. ReAct tool loop with a scripted fake LLM client
5. Error recovery (typo recovery in model code)
6. State rollback on unexpected exceptions
7. Data loader formats and unsupported file error
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from agent import DataAgent, SecurityException, validate_code_safety
from data_loader import load_table


class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, tool_id: str, name: str, arguments: dict):
        self.id = tool_id
        self.type = "function"
        self.function = MockFunction(name, json.dumps(arguments))


class MockMessage:
    def __init__(self, content: str = None, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message


class MockResponse:
    def __init__(self, message: MockMessage):
        self.choices = [MockChoice(message)]


class ScriptedLLMClient:
    """Scripted fake client mimicking openai.chat.completions.create."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            return MockResponse(MockMessage(content="No more scripted responses."))
        next_msg = self.responses.pop(0)
        return MockResponse(next_msg)


class TestToolsAndSafety(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame({
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-3"],
            "region": ["North", "South", "North", "North"],
            "quantity": [10, 20, 30, 30],
            "price": [100.0, 50.0, 75.0, 75.0],
            "rating": [5.0, np.nan, 4.0, 4.0],
        })
        self.agent = DataAgent(self.df, output_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_inspect_data(self):
        output = self.agent.inspect_data()
        self.assertIn("Shape: 4 rows, 5 columns", output)
        self.assertIn("Duplicate rows count: 1", output)
        self.assertIn("Missing Values per Column:", output)
        self.assertIn("rating: 1", output)
        self.assertIn("Numeric Summary:", output)
        self.assertIn("First 5 Rows:", output)

    def test_run_python_basic_execution(self):
        res = self.agent.run_python("total = df['quantity'].sum()\ntotal")
        self.assertIn("90", res)
        # Verify persistence of defined variable
        res2 = self.agent.run_python("total * 2")
        self.assertIn("180", res2)

    def test_run_python_security_blocked_import(self):
        for mod in ["os", "sys", "subprocess", "shutil", "socket"]:
            res = self.agent.run_python(f"import {mod}")
            self.assertIn("SecurityError", res)
            self.assertIn("blocked for safety", res)

        # Import from
        res_from = self.agent.run_python("from os import path")
        self.assertIn("SecurityError", res_from)

    def test_run_python_security_blocked_builtins(self):
        for b in ["open('test.txt')", "eval('1+1')", "exec('a=1')", "__import__('math')", "input()"]:
            res = self.agent.run_python(b)
            self.assertIn("SecurityError", res)

    def test_run_python_security_dunder_access(self):
        res = self.agent.run_python("().__class__.__bases__")
        self.assertIn("SecurityError", res)

    def test_run_python_plot_saving(self):
        code = (
            "plt.figure()\n"
            "plt.plot([1, 2, 3], [10, 20, 30])\n"
            "plt.title('Test Plot')\n"
            "plt.xlabel('X')\n"
            "plt.ylabel('Y')\n"
        )
        res = self.agent.run_python(code)
        self.assertIn("[Plot saved to", res)
        # Check that the file was created in the output directory
        png_files = list(Path(self.test_dir).glob("*.png"))
        self.assertEqual(len(png_files), 1)

    def test_run_python_error_traceback(self):
        res = self.agent.run_python("1 / 0")
        self.assertIn("Error:", res)
        self.assertIn("ZeroDivisionError", res)
        # Verify truncated traceback (does not contain internal runner frames)
        self.assertLess(len(res.splitlines()), 10)

    def test_run_python_output_truncation(self):
        # Generate string > 4000 characters
        code = "print('A' * 5000)"
        res = self.agent.run_python(code)
        self.assertIn("[Output truncated at 4000 characters]", res)
        self.assertLessEqual(len(res), 4100)


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame({
            "product": ["Widget A", "Widget B"],
            "sales": [150, 300],
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_loop_typo_recovery(self):
        """
        Simulate model making a code typo, receiving the error, fixing it,
        and answering accurately.
        """
        scripted_responses = [
            # Step 1: Model calls tool with a typo in column name ('revenue' instead of 'sales')
            MockMessage(
                tool_calls=[
                    MockToolCall(
                        "call_1",
                        "run_python",
                        {"code": "total = df['revenue'].sum()\ntotal"},
                    )
                ]
            ),
            # Step 2: Model receives KeyError, corrects column to 'sales'
            MockMessage(
                tool_calls=[
                    MockToolCall(
                        "call_2",
                        "run_python",
                        {"code": "total = df['sales'].sum()\ntotal"},
                    )
                ]
            ),
            # Step 3: Model receives 450, gives final answer
            MockMessage(
                content="The total sales across all products is 450."
            ),
        ]

        fake_client = ScriptedLLMClient(scripted_responses)
        agent = DataAgent(self.df, client=fake_client, output_dir=self.test_dir)

        result = agent.ask("What are total sales?")

        self.assertIn("450", result["answer"])
        self.assertEqual(len(result["tool_calls"]), 2)
        # First call has error
        self.assertIn("KeyError", result["tool_calls"][0]["result"])
        # Second call succeeded
        self.assertIn("450", result["tool_calls"][1]["result"])

    def test_loop_plot_generation(self):
        """Simulate model generating a chart and returning final answer."""
        scripted_responses = [
            MockMessage(
                tool_calls=[
                    MockToolCall(
                        "call_plot",
                        "run_python",
                        {
                            "code": "plt.figure()\nplt.bar(df['product'], df['sales'])\nplt.title('Sales')\n"
                        },
                    )
                ]
            ),
            MockMessage(
                content="Here is the bar chart showing sales for Widget A and Widget B."
            ),
        ]

        fake_client = ScriptedLLMClient(scripted_responses)
        agent = DataAgent(self.df, client=fake_client, output_dir=self.test_dir)

        result = agent.ask("Plot the sales by product.")

        self.assertIn("Widget A", result["answer"])
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertIn("[Plot saved to", result["tool_calls"][0]["result"])
        png_files = list(Path(self.test_dir).glob("*.png"))
        self.assertEqual(len(png_files), 1)

    def test_loop_blocked_import_in_loop(self):
        """Simulate model trying a blocked import and then responding politely."""
        scripted_responses = [
            MockMessage(
                tool_calls=[
                    MockToolCall(
                        "call_sec",
                        "run_python",
                        {"code": "import os\nos.getcwd()"},
                    )
                ]
            ),
            MockMessage(
                content="Operating system access is blocked for safety. I will analyze using pandas only."
            ),
        ]

        fake_client = ScriptedLLMClient(scripted_responses)
        agent = DataAgent(self.df, client=fake_client, output_dir=self.test_dir)

        result = agent.ask("Check working directory.")
        self.assertIn("blocked", result["answer"].lower())
        self.assertIn("SecurityError", result["tool_calls"][0]["result"])

    def test_history_rollback_on_exception(self):
        """Verify conversation history rollback when an unexpected exception occurs."""
        class FailingClient:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                raise RuntimeError("API connection failure")

        agent = DataAgent(self.df, client=FailingClient(), output_dir=self.test_dir)
        initial_msg_count = len(agent.messages)

        with self.assertRaises(RuntimeError):
            agent.ask("Should fail and rollback")

        # History should be rolled back to original state
        self.assertEqual(len(agent.messages), initial_msg_count)

    def test_max_step_limit(self):
        """Verify loop terminates politely when 10 steps are exceeded."""
        # Create an endless stream of tool calls
        infinite_tool_calls = [
            MockMessage(
                tool_calls=[
                    MockToolCall(
                        f"call_{i}",
                        "run_python",
                        {"code": f"x = {i}"},
                    )
                ]
            )
            for i in range(12)
        ]

        fake_client = ScriptedLLMClient(infinite_tool_calls)
        agent = DataAgent(self.df, client=fake_client, output_dir=self.test_dir)

        result = agent.ask("Endless question")
        self.assertIn("limit of 10 analytical steps", result["answer"])
        self.assertEqual(len(result["tool_calls"]), 10)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_csv(self):
        path = Path(self.test_dir) / "test.csv"
        path.write_text("a,b\n1,2\n3,4", encoding="utf-8")
        df = load_table(path)
        self.assertEqual(df.shape, (2, 2))

    def test_load_tsv(self):
        path = Path(self.test_dir) / "test.tsv"
        path.write_text("a\tb\n1\t2\n3\t4", encoding="utf-8")
        df = load_table(path)
        self.assertEqual(df.shape, (2, 2))

    def test_load_json(self):
        path = Path(self.test_dir) / "test.json"
        path.write_text('[{"a": 1, "b": 2}]', encoding="utf-8")
        df = load_table(path)
        self.assertEqual(df.shape, (1, 2))

    def test_load_parquet(self):
        path = Path(self.test_dir) / "test.parquet"
        orig_df = pd.DataFrame({"x": [10, 20]})
        orig_df.to_parquet(path)
        df = load_table(path)
        self.assertEqual(df.shape, (2, 1))

    def test_unsupported_format(self):
        path = Path(self.test_dir) / "test.docx"
        path.write_text("dummy", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_table(path)
        self.assertIn("Unsupported file format", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
