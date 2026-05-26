from sba_uas.compat.roach_policy import ensure_roach_on_path


def test_ensure_roach_on_path_does_not_import_carla():
    roach_dir = ensure_roach_on_path()

    assert roach_dir.name == "carla-roach"
    assert roach_dir.is_dir()
