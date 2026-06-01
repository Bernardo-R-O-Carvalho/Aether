"""
Aether Sandbox — Restricted Execution Environment
===================================================

What this does:
  AI-generated code runs here, not directly on the OS.
  The sandbox intercepts every action the code attempts and routes it
  through the Aether policy engine before execution.

  If the policy allows: action executes, result returned.
  If the policy blocks: action is rejected, reason logged.
  If risk exceeds threshold: model is frozen, all actions blocked.

Actions intercepted:
  - file.read      open(), read()
  - file.write     write(), writelines()
  - file.delete    os.remove(), os.unlink()
  - network.call   urllib, requests, http
  - syscall        os.system(), subprocess, exec
  - code.execute   eval(), exec(), compile()

Architecture:
  SandboxedCode          — wraps AI-generated code with a restricted namespace
  SandboxInterceptor     — intercepts and routes each action through policy
  SandboxResult          — result of running code in the sandbox

Usage:
  sandbox = AetherSandbox(policy)
  result = sandbox.run(ai_generated_code)
  print(result.output)
  print(result.blocked_actions)
"""

import sys
import io
import os
import ast
import traceback
from typing import Optional
from policy_engine import Policy, print_decision


# ─────────────────────────────────────────────
#  Sandbox result
# ─────────────────────────────────────────────

class SandboxResult:
    def __init__(self):
        self.output          = ""      # stdout captured
        self.return_value    = None    # return value if any
        self.error           = None    # exception if raised
        self.blocked_actions = []      # list of (action, param, reason)
        self.allowed_actions = []      # list of (action, param)
        self.success         = False

    def __repr__(self):
        lines = [f"SandboxResult(success={self.success})"]
        if self.output:
            lines.append(f"  output: {self.output[:100]!r}")
        if self.error:
            lines.append(f"  error:  {self.error}")
        if self.blocked_actions:
            lines.append(f"  blocked: {len(self.blocked_actions)} actions")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Restricted builtins and interceptors
# ─────────────────────────────────────────────

class BlockedAction(Exception):
    """Raised when a sandboxed action is blocked by policy."""
    def __init__(self, action: str, param: str, reason: str):
        self.action = action
        self.param  = param
        self.reason = reason
        super().__init__(f"Blocked: {action}({param})")


def make_restricted_open(policy: Policy, result: SandboxResult, allowed_path: str = "/sandbox"):
    """Create a restricted open() that routes through the policy."""
    real_open = open

    def restricted_open(path, mode="r", *args, **kwargs):
        path_str = str(path)
        if "w" in str(mode) or "a" in str(mode):
            action = "file.write"
        elif "x" in str(mode):
            action = "file.write"
        else:
            action = "file.read"

        decision, rule_hit, reason = policy.evaluate(action, path_str)
        print_decision(action, path_str, decision, reason)

        if decision == "allowed":
            result.allowed_actions.append((action, path_str))
            # In real sandbox, only allow reads from safe paths
            if action == "file.read" and path_str.startswith(allowed_path):
                # Return empty file for demo (no actual FS access)
                return io.StringIO(f"[sandboxed content of {path_str}]")
            else:
                return io.StringIO("")
        else:
            result.blocked_actions.append((action, path_str, reason))
            raise BlockedAction(action, path_str, reason)

    return restricted_open


def make_restricted_import(policy: Policy, result: SandboxResult):
    """Create a restricted __import__ that blocks dangerous modules."""
    BLOCKED_MODULES = {
        "os", "sys", "subprocess", "socket", "urllib",
        "requests", "http", "ftplib", "smtplib", "shutil",
        "pickle", "marshal", "ctypes", "importlib",
    }
    ALLOWED_MODULES = {
        "math", "random", "json", "re", "datetime",
        "collections", "itertools", "functools", "string",
        "time", "typing", "dataclasses", "enum",
    }

    real_import = __import__

    def restricted_import(name, *args, **kwargs):
        base_module = name.split(".")[0]
        if base_module in BLOCKED_MODULES:
            action = "syscall"
            param = f"import {name}"
            decision, rule_hit, reason = policy.evaluate(action, param)
            print_decision(action, param, decision, reason)
            result.blocked_actions.append((action, param, reason))
            raise BlockedAction(action, param, reason)
        return real_import(name, *args, **kwargs)

    return restricted_import


def make_blocked_exec(policy: Policy, result: SandboxResult):
    """Block eval() and exec() calls."""
    def blocked_eval(code, *args, **kwargs):
        action = "code.execute"
        param = str(code)[:50]
        decision, rule_hit, reason = policy.evaluate(action, param)
        print_decision(action, param, decision, reason)
        result.blocked_actions.append((action, param, reason))
        if decision != "allowed":
            raise BlockedAction(action, param, reason)
        return eval(code, *args, **kwargs)

    def blocked_exec(code, *args, **kwargs):
        action = "code.execute"
        param = str(code)[:50]
        decision, rule_hit, reason = policy.evaluate(action, param)
        print_decision(action, param, decision, reason)
        result.blocked_actions.append((action, param, reason))
        if decision != "allowed":
            raise BlockedAction(action, param, reason)
        exec(code, *args, **kwargs)

    return blocked_eval, blocked_exec


# ─────────────────────────────────────────────
#  Sandbox
# ─────────────────────────────────────────────

class AetherSandbox:
    """
    Restricted execution environment for AI-generated code.

    Every action the code attempts is routed through the Aether policy
    engine before execution. Blocked actions raise BlockedAction.
    All output is captured.

    Usage:
      sandbox = AetherSandbox(policy)
      result = sandbox.run(code_string)
    """
    def __init__(self, policy: Policy, allowed_path: str = "/sandbox"):
        self.policy       = policy
        self.allowed_path = allowed_path

    def run(self, code: str) -> SandboxResult:
        """
        Execute code in the sandbox.

        Returns SandboxResult with output, blocked actions, and status.
        """
        result = SandboxResult()

        # Capture stdout
        stdout_capture = io.StringIO()

        # Build restricted namespace
        restricted_open = make_restricted_open(self.policy, result, self.allowed_path)
        restricted_import = make_restricted_import(self.policy, result)
        blocked_eval, blocked_exec = make_blocked_exec(self.policy, result)

        safe_builtins = {
            # Safe builtins
            "print":       lambda *a, **k: print(*a, **k, file=stdout_capture),
            "len":         len,
            "range":       range,
            "enumerate":   enumerate,
            "zip":         zip,
            "map":         map,
            "filter":      filter,
            "sorted":      sorted,
            "reversed":    reversed,
            "list":        list,
            "dict":        dict,
            "set":         set,
            "tuple":       tuple,
            "str":         str,
            "int":         int,
            "float":       float,
            "bool":        bool,
            "type":        type,
            "isinstance":  isinstance,
            "hasattr":     hasattr,
            "getattr":     getattr,
            "setattr":     setattr,
            "min":         min,
            "max":         max,
            "sum":         sum,
            "abs":         abs,
            "round":       round,
            "repr":        repr,
            "format":      format,
            # Intercepted
            "open":        restricted_open,
            "__import__":  restricted_import,
            "eval":        blocked_eval,
            "exec":        blocked_exec,
            # Blocked
            "compile":     lambda *a, **k: (_ for _ in ()).throw(
                               BlockedAction("code.execute", "compile()", "compile() is not allowed")),
        }

        namespace = {"__builtins__": safe_builtins}

        try:
            # Validate syntax first
            ast.parse(code)
        except SyntaxError as e:
            result.error = f"Syntax error: {e}"
            return result

        try:
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            exec(code, namespace)
            sys.stdout = old_stdout
            result.success = True
        except BlockedAction as e:
            sys.stdout = old_stdout
            result.error = f"Blocked: {e.action}({e.param})"
        except Exception as e:
            sys.stdout = old_stdout
            result.error = f"{type(e).__name__}: {e}"

        result.output = stdout_capture.getvalue()
        return result

    def print_result(self, code: str, result: SandboxResult):
        """Print sandbox execution result in Aether's visual style."""
        print(f"\n  ⟁  Aether Sandbox — execution report")
        print(f"  {'─'*50}")
        print(f"  Status:          {'✓ completed' if result.success else '✗ blocked/error'}")
        if result.output:
            print(f"  Output:          {result.output.strip()[:100]!r}")
        if result.error:
            print(f"  Error:           {result.error}")
        print(f"  Allowed actions: {len(result.allowed_actions)}")
        print(f"  Blocked actions: {len(result.blocked_actions)}")
        if result.blocked_actions:
            print(f"  Blocked:")
            for action, param, reason in result.blocked_actions:
                print(f"    ✗  {action}({param[:40]})")
        print()
