def test_public_model_and_command_names():
    from hvl_interact.cli import parser
    from hvl_interact.models import HVLInteract

    assert parser().prog == "hvl-interact"
    assert HVLInteract.__name__ == "HVLInteract"
