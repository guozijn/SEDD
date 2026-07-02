from sedd_mini.mcqa_data import exact_choice_reward, extract_choice, format_arc_prompt


def test_format_arc_prompt_and_choice_reward():
    prompt, labels = format_arc_prompt(
        "Which surface produces the most heat?",
        {"label": ["A", "B"], "text": ["dry palms", "wet palms"]},
    )

    assert "Question: Which surface produces the most heat?" in prompt
    assert "A. dry palms" in prompt
    assert labels == ["A", "B"]
    assert extract_choice("Answer: b", labels) == "B"
    assert exact_choice_reward("Answer: B", "B", labels) == 1.0
    assert exact_choice_reward("Answer: A", "B", labels) == 0.0
