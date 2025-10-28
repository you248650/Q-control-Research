def test_import():
    import qpulse_nav
    assert hasattr(qpulse_nav, "run")
