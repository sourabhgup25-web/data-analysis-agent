"""
Data Analysis Agent core module.

Implements the DataAgent class featuring:
- ReAct-style tool-calling loop with OpenAI-compatible API
- General data inspection (inspect_data)
- Python REPL execution (run_python) against the loaded DataFrame 'df'
- AST-based safety validation
- Matplotlib figure auto-saving with custom theme
- Full conversation memory with state rollback on error
- CLI interactive chat mode
"""

import ast
import contextlib
import copy
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for scripts and servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from data_loader import load_table

# Load environment variables from .env if present
load_dotenv()

# Palette & plot styling based on design requirements
CHART_COLORS = ["#14213D", "#B08D57", "#4F6D8F", "#7A9A8A", "#8B2E3C"]


def apply_plot_style():
    """Apply refined plot styling with coordinated palette and clean layout."""
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=CHART_COLORS),
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "figure.autolayout": True,
    })


apply_plot_style()


class SecurityException(Exception):
    """Raised when unsafe code patterns are detected by the AST guard."""
    pass


class CodeSecurityGuard(ast.NodeVisitor):
    """
    Lightweight AST guard to block unauthorized modules, builtins, and dunder attributes.
    NOTE: This is a lightweight filter, NOT a secure operating-system level sandbox.
    """
    BLOCKED_MODULES = {"os", "sys", "subprocess", "shutil", "socket"}
    BLOCKED_BUILTINS = {"open", "eval", "exec", "compile", "__import__", "input"}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_pkg = alias.name.split(".")[0]
            if base_pkg in self.BLOCKED_MODULES:
                raise SecurityException(f"Importing module '{alias.name}' is blocked for safety.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_pkg = node.module.split(".")[0]
            if base_pkg in self.BLOCKED_MODULES:
                raise SecurityException(f"Importing from module '{node.module}' is blocked for safety.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self.BLOCKED_BUILTINS:
            raise SecurityException(f"Use of built-in function '{node.func.id}' is blocked for safety.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.BLOCKED_BUILTINS and isinstance(node.ctx, ast.Load):
            # Check for referencing dangerous builtins directly
            raise SecurityException(f"Access to '{node.id}' is blocked for safety.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise SecurityException(f"Access to dunder attribute '{node.attr}' is blocked for safety.")
        self.generic_visit(node)


def validate_code_safety(code: str):
    """Parse and validate python code against security rules."""
    tree = ast.parse(code)
    guard = CodeSecurityGuard()
    guard.visit(tree)


SYSTEM_PROMPT = """You are an expert Data Analysis Agent. You assist users by querying, exploring, and analyzing tabular datasets.

Rules:
1. Never invent or hallucinate numbers. Every single figure, count, statistic, or calculation must be computed using your tools.
2. Inspect the dataset first using inspect_data() unless the schema and column details are already known.
3. Check data quality when relevant (missing values, duplicates, outliers, inconsistent labels, wrong dtypes) and explicitly report anything that affects the analysis.
4. When plotting charts using run_python, always label the title, x-axis, and y-axis clearly.
5. Format your final answers following this structure:
   - Direct answer first.
   - Supporting numbers and computations.
   - Assumptions, caveats, and data quality notes.
6. If the data cannot answer the question (for example, calculating profit or profit margin when no cost column exists), clearly state that the dataset cannot answer it rather than guessing.
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_data",
            "description": "Inspect the dataset: returns shape, duplicate count, dtypes, missing values per column, unique counts, numeric summary, top values of text columns, and the first 5 rows.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code against the pandas DataFrame 'df'. Standard libraries available: pd (pandas), np (numpy), plt (matplotlib.pyplot). Variables persist across calls. Matplotlib charts are automatically saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute against df.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


class DataAgent:
    """
    ReAct-style data analysis agent interacting with an OpenAI-compatible API.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        output_dir: str = "outputs",
    ):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("LLM_API_KEY", "dummy-key")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        # Allow passing an injected client (useful for offline testing)
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # REPL execution namespace; variables persist across run_python invocations
        self.locals: Dict[str, Any] = {
            "df": self.df,
            "pd": pd,
            "np": np,
            "plt": plt,
        }

        # Conversation history initialized with system prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self.step_history: List[Dict[str, Any]] = []

    def inspect_data(self) -> str:
        """Inspect the DataFrame and return structural and statistical overview."""
        df = self.df
        lines = []
        lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        lines.append(f"Duplicate rows count: {df.duplicated().sum()}")
        lines.append("\nColumn Data Types:")
        for col, dtype in df.dtypes.items():
            lines.append(f"  - {col}: {dtype}")

        missing = df.isnull().sum()
        lines.append("\nMissing Values per Column:")
        for col, count in missing.items():
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            lines.append(f"  - {col}: {count} ({pct:.1f}%)")

        lines.append("\nUnique Values per Column:")
        for col in df.columns:
            lines.append(f"  - {col}: {df[col].nunique()} unique")

        # Summary of numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            lines.append("\nNumeric Summary:")
            lines.append(df[num_cols].describe().to_string())

        # Top values for categorical/text columns
        text_cols = df.select_dtypes(include=["object", "str", "category"]).columns
        if len(text_cols) > 0:
            lines.append("\nTop Values for Text/Categorical Columns:")
            for col in text_cols[:8]:
                top_vals = df[col].value_counts().head(3).to_dict()
                lines.append(f"  - {col}: {top_vals}")

        lines.append("\nFirst 5 Rows:")
        lines.append(df.head(5).to_string())

        return "\n".join(lines)

    def run_python(self, code: str) -> str:
        """
        Execute python code in REPL style against self.locals.
        Captures stdout and the evaluated value of the final expression.
        Saves any generated matplotlib figures.
        """
        # 1. AST Safety Check
        try:
            validate_code_safety(code)
        except SecurityException as e:
            return f"SecurityError: {e}"
        except SyntaxError as e:
            return f"SyntaxError during security validation: {e}"

        # 2. Parse code for REPL execution
        try:
            parsed = ast.parse(code)
        except SyntaxError as e:
            tb = traceback.format_exc().strip().splitlines()
            return "SyntaxError:\n" + "\n".join(tb[-4:])

        stdout_buf = io.StringIO()
        saved_plots = []
        eval_result = None

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                if not parsed.body:
                    return ""

                # Check if the final node is an expression
                last_node = parsed.body[-1]
                if isinstance(last_node, ast.Expr):
                    # Execute all preceding statements
                    if len(parsed.body) > 1:
                        exec_mod = ast.Module(body=parsed.body[:-1], type_ignores=[])
                        compiled_exec = compile(exec_mod, filename="<agent_code>", mode="exec")
                        exec(compiled_exec, self.locals, self.locals)

                    # Evaluate the last expression
                    eval_mod = ast.Expression(body=last_node.value)
                    compiled_eval = compile(eval_mod, filename="<agent_code>", mode="eval")
                    eval_result = eval(compiled_eval, self.locals, self.locals)
                else:
                    # Execute all statements
                    compiled = compile(parsed, filename="<agent_code>", mode="exec")
                    exec(compiled, self.locals, self.locals)

                # Check if matplotlib created any figures
                fig_nums = plt.get_fignums()
                if fig_nums:
                    for fig_num in fig_nums:
                        fig = plt.figure(fig_num)
                        timestamp = int(time.time() * 1000)
                        plot_filename = f"plot_{timestamp}_{fig_num}.png"
                        plot_path = self.output_dir / plot_filename
                        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                        saved_plots.append(str(plot_path))
                    plt.close("all")

        except Exception:
            tb = traceback.format_exc().strip().splitlines()
            # Return only the last few lines of the traceback
            error_msg = "Error:\n" + "\n".join(tb[-4:])
            return error_msg

        # Assemble output
        output_parts = []
        printed = stdout_buf.getvalue().strip()
        if printed:
            output_parts.append(printed)

        if eval_result is not None:
            # Format dataframe or object representation
            if isinstance(eval_result, pd.DataFrame):
                output_parts.append(eval_result.to_string())
            elif isinstance(eval_result, pd.Series):
                output_parts.append(eval_result.to_string())
            else:
                output_parts.append(repr(eval_result))

        if saved_plots:
            for p in saved_plots:
                output_parts.append(f"[Plot saved to {p}]")

        full_output = "\n".join(output_parts).strip()
        if not full_output:
            full_output = "Code executed successfully with no output."

        # Truncate long outputs to ~4000 characters
        if len(full_output) > 4000:
            full_output = full_output[:4000] + "\n... [Output truncated at 4000 characters]"

        return full_output

    def ask(
        self,
        question: str,
        on_tool_call: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ask a question in natural language.
        Runs the ReAct loop up to 10 steps, rolling back history on failure.

        Parameters:
            question: The user prompt.
            on_tool_call: Optional callback: fn(tool_name, arguments, tool_result)

        Returns:
            Dictionary with keys:
                - 'answer': The final textual response
                - 'tool_calls': List of tool calls executed during this question
        """
        # Save history checkpoint for rollback on exception
        history_snapshot = copy.deepcopy(self.messages)

        current_tool_calls_record = []
        self.messages.append({"role": "user", "content": question})

        try:
            steps = 0
            max_steps = 10

            while steps < max_steps:
                steps += 1

                # Send conversation plus tool definitions to LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                )

                choice = response.choices[0]
                message = choice.message

                # Convert message object to dictionary for message history
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                }

                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Model requested one or more tool calls
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                    self.messages.append(assistant_msg)

                    for tc in message.tool_calls:
                        tool_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {"raw": tc.function.arguments}

                        # Execute tool
                        if tool_name == "inspect_data":
                            result = self.inspect_data()
                        elif tool_name == "run_python":
                            code_to_run = args.get("code", "")
                            result = self.run_python(code_to_run)
                        else:
                            result = f"Unknown tool: '{tool_name}'."

                        record_entry = {
                            "step": steps,
                            "tool": tool_name,
                            "arguments": args,
                            "result": result,
                        }
                        current_tool_calls_record.append(record_entry)

                        if on_tool_call:
                            on_tool_call(tool_name, args, result)

                        # Append tool response
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        })

                    # Continue loop to give model the tool results
                    continue

                else:
                    # Model answered directly without tool calls
                    self.messages.append(assistant_msg)
                    final_answer = message.content or ""
                    return {
                        "answer": final_answer,
                        "tool_calls": current_tool_calls_record,
                    }

            # Loop reached 10 steps without terminating
            limit_msg = (
                "I have reached the limit of 10 analytical steps for this question without "
                "concluding. Please refine or break down your question so I can assist you better."
            )
            self.messages.append({"role": "assistant", "content": limit_msg})
            return {
                "answer": limit_msg,
                "tool_calls": current_tool_calls_record,
            }

        except Exception as e:
            # Roll back conversation history to state before this question
            self.messages = history_snapshot
            raise e


def terminal_chat(file_path: str):
    """Run an interactive CLI chat session against a tabular dataset."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found at '{file_path}'")
        sys.exit(1)

    print(f"Loading {file_path}...")
    df = load_table(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    agent = DataAgent(df)
    print("Data Analysis Agent ready. Type 'exit' or 'quit' to end.\n")

    def print_tool(name: str, args: Dict[str, Any], result: str):
        print(f"\n[Tool Execution: {name}]")
        if "code" in args:
            print("--- Code ---")
            print(args["code"])
        print("--- Output ---")
        print(result)
        print("-------------\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            response = agent.ask(user_input, on_tool_call=print_tool)
            print(f"\nAgent:\n{response['answer']}")

        except KeyboardInterrupt:
            print("\nExiting session.")
            break
        except Exception as err:
            print(f"\nError encountered: {err}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        terminal_chat(sys.argv[1])
    else:
        print("Usage: python agent.py <path_to_dataset>")
