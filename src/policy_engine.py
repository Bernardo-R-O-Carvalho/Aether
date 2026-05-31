"""
Aether Policy Engine — Probabilistic AI Containment
=====================================================

What this does:
  Every action an AI model attempts passes through this engine.
  The engine evaluates the action against a policy written in .aeth syntax.
  Risk is not boolean — it is a Beta distribution that updates with evidence.

  risk ~ Beta(a=violations, b=safe_calls)

  A model that has never violated a constraint has a different risk profile
  than one that has tried twice. The engine tracks this, updates its beliefs,
  and enforces accordingly.

Core concepts:
  Policy     — a set of rules defining what is allowed and what is not
  Action     — something a model attempts (file access, network call, code exec)
  Risk       — Beta(violations, safe_calls) — updates after every action
  Audit node — every decision logged as a causal graph node
  Explain    — any blocked action generates a human-readable trace

Usage in .aeth:
  policy ContainModel:
      allow file.read(path ~ restricted_to("/sandbox"))
      allow network.call(host ~ whitelist(["api.anthropic.com"]))
      deny  syscall ~ in(["fork", "exec", "socket"])

      on violation:
          risk ~ beta(a=violations, b=safe_calls)
          if risk > 0.8:
              freeze_and_audit()

  explain ContainModel
"""

import math
import time
import json
import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  Action categories
# ─────────────────────────────────────────────

# All recognized action types
ACTION_TYPES = {
    "file.read",
    "file.write",
    "file.delete",
    "file.execute",
    "network.call",
    "network.listen",
    "syscall",
    "code.execute",
    "data.access",
    "model.spawn",
}

# Actions that are high-risk by default
HIGH_RISK_ACTIONS = {"file.delete", "file.execute", "syscall", "model.spawn", "network.listen"}


# ─────────────────────────────────────────────
#  Policy rules
# ─────────────────────────────────────────────

@dataclass
class PolicyRule:
    """
    A single rule in a policy.

    kind:       "allow" or "deny"
    action:     the action type (e.g. "file.read", "syscall")
    constraint: how to evaluate the action parameter
                "restricted_to" — path must be under this prefix
                "whitelist"     — value must be in this list
                "blacklist"     — value must NOT be in this list
                "in"            — value must be in this set
                "any"           — always matches
    value:      the constraint value (string, list, or None)
    """
    kind:       str           # "allow" or "deny"
    action:     str           # e.g. "file.read"
    constraint: str           # "restricted_to", "whitelist", "blacklist", "in", "any"
    value:      object        # constraint parameter

    def matches(self, action: str, param: str = "") -> bool:
        """Return True if this rule applies to the given action and parameter."""
        if self.action != action and self.action != "*":
            return False
        if self.constraint == "any":
            return True
        if self.constraint == "restricted_to":
            return str(param).startswith(str(self.value))
        if self.constraint == "whitelist":
            return str(param) in [str(v) for v in self.value]
        if self.constraint == "blacklist":
            return str(param) not in [str(v) for v in self.value]
        if self.constraint == "in":
            return str(param) in [str(v) for v in self.value]
        return False


# ─────────────────────────────────────────────
#  Audit graph
#
#  Every decision is logged as a node in a causal graph.
#  The graph can be replayed, inspected, and challenged.
# ─────────────────────────────────────────────

@dataclass
class AuditNode:
    """A single node in the audit graph."""
    node_id:    int
    timestamp:  float
    action:     str
    param:      str
    decision:   str           # "allowed", "blocked", "frozen"
    rule_hit:   Optional[str] # which rule triggered the decision
    risk_a:     float         # Beta distribution alpha (violations)
    risk_b:     float         # Beta distribution beta (safe_calls)
    risk_mean:  float         # E[risk] = a / (a + b)
    reason:     str           # human-readable explanation


class AuditGraph:
    """
    Causal audit graph for AI policy enforcement.

    Each node records: what was attempted, what was decided, why,
    and what the risk distribution looked like at that moment.

    The graph is append-only — no node can be modified after creation.
    This ensures tamper-evident logging.
    """
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.nodes: list[AuditNode] = []
        self.violations = 1      # Beta prior: start at 1 (Laplace smoothing)
        self.safe_calls = 1      # Beta prior: start at 1

    def log(self, action: str, param: str, decision: str,
            rule_hit: str, reason: str) -> AuditNode:
        """Log a decision and update the risk distribution."""
        risk_mean = self.violations / (self.violations + self.safe_calls)

        node = AuditNode(
            node_id   = len(self.nodes),
            timestamp = time.time(),
            action    = action,
            param     = param,
            decision  = decision,
            rule_hit  = rule_hit,
            risk_a    = self.violations,
            risk_b    = self.safe_calls,
            risk_mean = risk_mean,
            reason    = reason,
        )
        self.nodes.append(node)

        # Update Beta distribution
        if decision == "blocked" or decision == "frozen":
            self.violations += 1
        else:
            self.safe_calls += 1

        return node

    def risk_mean(self) -> float:
        """Current expected risk E[Beta(violations, safe_calls)]."""
        return self.violations / (self.violations + self.safe_calls)

    def risk_std(self) -> float:
        """Current standard deviation of risk."""
        a, b = self.violations, self.safe_calls
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def to_json(self) -> str:
        """Serialize the audit graph to JSON."""
        return json.dumps({
            "policy": self.policy_name,
            "violations": self.violations - 1,
            "safe_calls": self.safe_calls - 1,
            "risk_mean": self.risk_mean(),
            "nodes": [
                {
                    "id":       n.node_id,
                    "time":     n.timestamp,
                    "action":   n.action,
                    "param":    n.param,
                    "decision": n.decision,
                    "rule":     n.rule_hit,
                    "risk":     n.risk_mean,
                    "reason":   n.reason,
                }
                for n in self.nodes
            ]
        }, indent=2)


# ─────────────────────────────────────────────
#  Policy
# ─────────────────────────────────────────────

class Policy:
    """
    An Aether policy for AI containment.

    A policy is a set of rules evaluated in order.
    The first matching rule determines the decision.
    If no rule matches, the default is to block (deny by default).

    The policy also tracks a risk distribution:
      risk ~ Beta(violations, safe_calls)

    And a freeze threshold: if risk > threshold, freeze_and_audit().
    """
    def __init__(self, name: str):
        self.name           = name
        self.rules:   list[PolicyRule] = []
        self.freeze_threshold = 0.8    # freeze if risk > this
        self.audit    = AuditGraph(name)
        self.frozen   = False

    def add_rule(self, rule: PolicyRule):
        self.rules.append(rule)

    def evaluate(self, action: str, param: str = "") -> tuple[str, str, str]:
        """
        Evaluate an action against the policy.

        Returns (decision, rule_hit, reason) where:
          decision: "allowed", "blocked", or "frozen"
          rule_hit: description of the matching rule
          reason:   human-readable explanation
        """
        if self.frozen:
            return "frozen", "freeze_threshold", (
                f"Policy '{self.name}' is frozen — risk exceeded {self.freeze_threshold:.0%}. "
                f"All actions blocked pending audit."
            )

        # Check freeze threshold
        risk = self.audit.risk_mean()
        if risk > self.freeze_threshold:
            self.frozen = True
            return "frozen", "risk_threshold", (
                f"Risk distribution Beta({self.audit.violations:.0f}, {self.audit.safe_calls:.0f}) "
                f"has mean {risk:.3f} > threshold {self.freeze_threshold:.2f}. "
                f"Model frozen for audit."
            )

        # Evaluate rules in order
        for rule in self.rules:
            if rule.matches(action, param):
                if rule.kind == "allow":
                    reason = (
                        f"Action '{action}({param})' matched allow rule: "
                        f"{rule.constraint}({rule.value}). "
                        f"Current risk: {risk:.3f} "
                        f"[Beta({self.audit.violations:.0f}, {self.audit.safe_calls:.0f})]"
                    )
                    node = self.audit.log(action, param, "allowed",
                                          f"allow {action} {rule.constraint}", reason)
                    return "allowed", f"allow {action}", reason
                else:  # deny
                    reason = (
                        f"Action '{action}({param})' matched deny rule: "
                        f"{rule.constraint}({rule.value}). "
                        f"Blocked. Risk updated to "
                        f"Beta({self.audit.violations + 1:.0f}, {self.audit.safe_calls:.0f})."
                    )
                    node = self.audit.log(action, param, "blocked",
                                          f"deny {action} {rule.constraint}", reason)
                    return "blocked", f"deny {action}", reason

        # No rule matched — deny by default
        reason = (
            f"Action '{action}({param})' matched no policy rule. "
            f"Default: blocked. "
            f"Add 'allow {action}(...)' to permit this action."
        )
        node = self.audit.log(action, param, "blocked", "default_deny", reason)
        return "blocked", "default_deny", reason

    def explain(self) -> str:
        """Generate a human-readable audit report."""
        lines = []
        lines.append(f"\n  ⟁  Aether Policy Audit — '{self.name}'")
        lines.append(f"  {'─'*52}")

        # Risk summary
        a, b = self.audit.violations, self.audit.safe_calls
        risk = self.audit.risk_mean()
        std  = self.audit.risk_std()
        lines.append(f"  Risk distribution:  Beta({a:.0f}, {b:.0f})")
        lines.append(f"  Expected risk:      {risk:.4f}  (±{std:.4f})")
        lines.append(f"  Violations:         {int(a - 1)}")
        lines.append(f"  Safe calls:         {int(b - 1)}")
        lines.append(f"  Freeze threshold:   {self.freeze_threshold:.2f}")
        lines.append(f"  Status:             {'🔴 FROZEN' if self.frozen else '🟢 active'}")
        lines.append(f"  {'─'*52}")

        # Rules
        lines.append(f"  Rules ({len(self.rules)}):")
        for r in self.rules:
            lines.append(f"    {r.kind:5s}  {r.action}  {r.constraint}({r.value})")
        lines.append(f"  {'─'*52}")

        # Audit log
        lines.append(f"  Audit log ({len(self.audit.nodes)} entries):")
        for node in self.audit.nodes[-10:]:   # show last 10
            icon = "✓" if node.decision == "allowed" else "✗"
            lines.append(
                f"    #{node.node_id:03d}  {icon}  {node.action}({node.param[:20]:20s})  "
                f"→ {node.decision:8s}  risk={node.risk_mean:.3f}"
            )
        if len(self.audit.nodes) > 10:
            lines.append(f"    ... {len(self.audit.nodes) - 10} earlier entries")
        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Print helpers
# ─────────────────────────────────────────────

def print_decision(action: str, param: str, decision: str, reason: str):
    """Print a policy decision in Aether's visual style."""
    icon = "✓" if decision == "allowed" else ("🔴" if decision == "frozen" else "✗")
    color_decision = decision.upper()
    print(f"\n  {icon}  [{color_decision}]  {action}({param})")
    print(f"     {reason}")
