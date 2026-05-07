from __future__ import annotations


def test_dispatch_delegate_task_forwards_bucket_b_omo_fields(monkeypatch):
    from run_agent import AIAgent
    import tools.delegate_tool as delegate_tool

    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return '{"results": []}'

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    agent = object.__new__(AIAgent)

    args = {
        "goal": "do work",
        "context": "ctx",
        "toolsets": ["terminal"],
        "tasks": None,
        "max_iterations": 3,
        "acp_command": "codex",
        "acp_args": ["--acp", "--stdio"],
        "role": "orchestrator",
        "agent": "atlas",
        "subagent_type": "builder-agent",
        "category": "implementation",
        "archetype": "implementer",
        "specialist": "builder",
        "route_category": "deep",
        "delegation_profile": "implementation",
        "runtime_mode": "execution_supervisor",
        "skills": ["test-driven-development"],
        "task_contract": {
            "task": "do work",
            "expected_outcome": "done",
            "required_skills": [],
            "required_tools": [],
            "must_do": ["work"],
            "must_not_do": ["mutate live"],
            "context": {"repo": "candidate"},
        },
        "named_workflow": {
            "schema": "hermes/named-workflow",
            "schema_version": "1.0",
            "workflow_name": "deep_worker",
            "mode": "execute",
            "objective": "do work",
            "plan": ["work"],
            "acceptance": ["done"],
            "taxonomy": {
                "named_workflow": "deep_worker",
                "workflow": "deep_worker",
                "specialist": "builder",
                "archetype": "implementer",
                "route_category": "deep",
                "runtime_mode": "execution_supervisor",
                "delegation_profile": "implementation",
            },
            "execution_task_contract": {
                "task": "do work",
                "expected_outcome": "done",
                "required_skills": [],
                "required_tools": [],
                "must_do": ["work"],
                "must_not_do": ["mutate live"],
                "context": {"repo": "candidate"},
            },
            "consumption": {"downstream_role": "worker"},
        },
    }

    result = agent._dispatch_delegate_task(args)

    assert result == '{"results": []}'
    for key, value in args.items():
        assert captured[key] == value
    assert captured["parent_agent"] is agent


def test_child_prompt_includes_wave1_overlay_header():
    from tools.delegate_tool import _build_child_system_prompt

    prompt = _build_child_system_prompt(
        "do work",
        "ctx",
        workspace_path="/tmp/example",
        wave1_overlay_prompt="## OMO / Named-Agent Runtime Contract\n{}",
    )

    assert "WAVE 1 DELEGATION INPUTS" in prompt
    assert "OMO / Named-Agent Runtime Contract" in prompt
