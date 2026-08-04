"""Guard tests: clean tree passes, dirty fixture and bad features fail."""

import os

from inference.meta import guard
from inference.meta.policy import POLICY_INPUT_FEATURES

META_DIR = os.path.join(os.path.dirname(guard.__file__))
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "dirty_meta")


def test_real_tree_is_clean():
    assert guard.run_guard(META_DIR, POLICY_INPUT_FEATURES) == []
    assert guard.main([]) == 0


def test_policy_features_are_whitelisted():
    assert guard.check_feature_names(POLICY_INPUT_FEATURES) == []


def test_dirty_fixture_fails_source_scan():
    violations = guard.scan_sources(FIXTURE_DIR)
    tokens_hit = {tok for tok in guard.FORBIDDEN_TOKENS
                  if any(f"'{tok}'" in v for v in violations)}
    assert tokens_hit == set(guard.FORBIDDEN_TOKENS)
    assert guard.main(["--root", FIXTURE_DIR]) == 1


def test_non_whitelisted_feature_fails():
    bad = list(POLICY_INPUT_FEATURES) + ["board_hash_prefix"]
    assert guard.check_feature_names(bad) != []


def test_pragma_line_is_skipped():
    assert "guard-ok" in guard.PRAGMA
