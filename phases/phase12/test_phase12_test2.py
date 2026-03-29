#!/usr/bin/env python3
"""
test_phase12_test2.py — Test: LLM Action Parsing

Tests:
1. Explicit ACTION: format parsing
2. Choice pattern parsing
3. Movement pattern parsing
4. Fuzzy matching fallbacks
5. Click coordinate extraction
6. Parse reliability metrics
"""

import sys
from pathlib import Path

# Add project root and phase12
PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR))

from flux_utils import PhaseResults


def test_explicit_format():
    """Test explicit ACTION: format parsing."""
    from action_parser import ActionParser, ACTION_MAP
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        ("I analyzed the puzzle. ACTION: UP", 1, "UP"),
        ("Looking at the grid, I'll go down. ACTION: DOWN", 2, "DOWN"),
        ("ACTION: LEFT is the best choice.", 3, "LEFT"),
        ("The goal is right. ACTION: RIGHT", 4, "RIGHT"),
        ("Let me interact. ACTION: INTERACT", 5, "INTERACT"),
        ("ACTION: UNDO to reverse.", 7, "UNDO"),
    ]
    
    for response, expected_id, expected_name in test_cases:
        total += 1
        result = parser.parse(response)
        
        if result.action_id == expected_id:
            print(f"  ✓ '{response[:30]}...' → {result.action_name}")
            passed += 1
        else:
            print(f"  ✗ Expected {expected_name}, got {result.action_name}")
    
    return passed, total


def test_choice_patterns():
    """Test 'I choose X' and similar patterns."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        ("I'll go up to explore.", 1),
        ("I will move down.", 2),
        ("I choose left because of the pattern.", 3),
        ("I select right as my action.", 4),
        ("I should go up to find the target.", 1),
    ]
    
    for response, expected_id in test_cases:
        total += 1
        result = parser.parse(response)
        
        if result.action_id == expected_id:
            print(f"  ✓ '{response[:35]}...' → {result.action_name}")
            passed += 1
        else:
            print(f"  ✗ Expected action {expected_id}, got {result.action_id}")
    
    return passed, total


def test_movement_patterns():
    """Test 'move X' and 'go X' patterns."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        ("Move up to see what's there.", 1),
        ("Go down the corridor.", 2),
        ("Head left toward the goal.", 3),
        ("Proceed right along the path.", 4),
    ]
    
    for response, expected_id in test_cases:
        total += 1
        result = parser.parse(response)
        
        if result.action_id == expected_id:
            print(f"  ✓ '{response[:35]}...' → {result.action_name}")
            passed += 1
        else:
            print(f"  ✗ Expected action {expected_id}, got {result.action_id}")
    
    return passed, total


def test_fuzzy_matching():
    """Test fuzzy matching when patterns are less clear."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        ("The best option is to go UP.", 1),
        ("DOWN seems like a good choice.", 2),
        ("I think LEFT will work.", 3),
        ("RIGHT!", 4),
    ]
    
    for response, expected_id in test_cases:
        total += 1
        result = parser.parse(response)
        
        if result.action_id == expected_id:
            print(f"  ✓ '{response[:35]}' → {result.action_name} (conf={result.confidence:.2f})")
            passed += 1
        else:
            print(f"  ✗ Expected action {expected_id}, got {result.action_id}")
    
    return passed, total


def test_click_coordinates():
    """Test CLICK action with coordinates."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        ("CLICK at (10, 20) to select the cell.", 6, (10, 20)),
        ("CLICK(5, 15) is the target.", 6, (5, 15)),
        ("I'll CLICK at 3, 7 here.", 6, (3, 7)),
    ]
    
    available = [1, 2, 3, 4, 5, 6, 7]  # Include CLICK
    
    for response, expected_id, expected_coords in test_cases:
        total += 1
        result = parser.parse(response, available)
        
        if result.action_id == expected_id and result.click_coords == expected_coords:
            print(f"  ✓ '{response[:30]}...' → CLICK @ {result.click_coords}")
            passed += 1
        elif result.action_id == expected_id:
            print(f"  ⚠ Got CLICK but coords wrong: {result.click_coords} vs {expected_coords}")
            passed += 0.5
        else:
            print(f"  ✗ Expected CLICK, got {result.action_name}")
    
    return int(passed), total


def test_available_actions_filter():
    """Test that parser respects available actions."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    # Test with limited actions
    total += 1
    available = [1, 2]  # Only UP and DOWN
    result = parser.parse("I'll go LEFT.", available)
    
    if result.action_id in available:
        print(f"  ✓ Respects available actions: got {result.action_name}")
        passed += 1
    else:
        print(f"  ✗ Should only return available actions, got {result.action_id}")
    
    # Test with single action
    total += 1
    available = [4]  # Only RIGHT
    result = parser.parse("I'll go UP.", available)
    
    if result.action_id == 4:
        print(f"  ✓ Falls back to only available action: RIGHT")
        passed += 1
    else:
        print(f"  ✗ Should return RIGHT, got {result.action_id}")
    
    return passed, total


def test_fallback_behavior():
    """Test fallback behavior for unparseable responses."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    passed = 0
    total = 0
    
    test_cases = [
        "",  # Empty
        "I'm not sure what to do...",  # No action mention
        "This is completely random text.",
    ]
    
    for response in test_cases:
        total += 1
        result = parser.parse(response, [1, 2, 3, 4])
        
        # Should return some valid action
        if result.action_id in [1, 2, 3, 4]:
            print(f"  ✓ Fallback works: '{response[:30]}...' → {result.action_name}")
            passed += 1
        else:
            print(f"  ✗ Invalid fallback action: {result.action_id}")
    
    return passed, total


def test_parse_reliability():
    """Test overall parse reliability across sample responses."""
    from action_parser import ActionParser
    
    parser = ActionParser()
    
    # Simulate 50 LLM-like responses
    sample_responses = [
        "Looking at the grid, I see a path going up. ACTION: UP",
        "The goal seems to be to the right. I'll move RIGHT.",
        "DOWN is clearly the way to go here.",
        "I choose to go LEFT based on the pattern.",
        "Let me try UP to see what's there.",
        "Moving DOWN would avoid the obstacle.",
        "LEFT LEFT LEFT! Go LEFT!",
        "The optimal action is to proceed RIGHT.",
        "ACTION: INTERACT to toggle the switch.",
        "I'll UNDO my last move.",
    ] * 5  # 50 total
    
    correct = 0
    total = len(sample_responses)
    
    for response in sample_responses:
        result = parser.parse(response, [1, 2, 3, 4, 5, 7])
        if result.confidence >= 0.5:
            correct += 1
    
    reliability = correct / total
    print(f"  Parse reliability: {correct}/{total} = {reliability:.1%}")
    print(f"  Threshold: >90%")
    
    return 1 if reliability > 0.9 else 0, 1


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Phase 12 Test 2: LLM Action Parsing")
    print("=" * 60 + "\n")
    
    results = PhaseResults(phase=12, component_name="Action Parser")
    
    total_passed = 0
    total_tests = 0
    
    # Run test suites
    print("Testing explicit ACTION: format...")
    passed, total = test_explicit_format()
    total_passed += passed
    total_tests += total
    results.add_test("Explicit Format", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.9)
    
    print("\nTesting choice patterns...")
    passed, total = test_choice_patterns()
    total_passed += passed
    total_tests += total
    results.add_test("Choice Patterns", passed=passed >= total * 0.8,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting movement patterns...")
    passed, total = test_movement_patterns()
    total_passed += passed
    total_tests += total
    results.add_test("Movement Patterns", passed=passed >= total * 0.8,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting fuzzy matching...")
    passed, total = test_fuzzy_matching()
    total_passed += passed
    total_tests += total
    results.add_test("Fuzzy Matching", passed=passed >= total * 0.6,
                     score=passed/total if total > 0 else 0, threshold=0.6)
    
    print("\nTesting click coordinates...")
    passed, total = test_click_coordinates()
    total_passed += passed
    total_tests += total
    results.add_test("Click Coordinates", passed=passed >= total * 0.5,
                     score=passed/total if total > 0 else 0, threshold=0.5)
    
    print("\nTesting available actions filter...")
    passed, total = test_available_actions_filter()
    total_passed += passed
    total_tests += total
    results.add_test("Available Filter", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=1.0)
    
    print("\nTesting fallback behavior...")
    passed, total = test_fallback_behavior()
    total_passed += passed
    total_tests += total
    results.add_test("Fallback", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=1.0)
    
    print("\nTesting parse reliability...")
    passed, total = test_parse_reliability()
    total_passed += passed
    total_tests += total
    results.add_test("Reliability >90%", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=1.0)
    
    # Summary
    print("\n" + "-" * 40)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    overall = total_passed / total_tests if total_tests > 0 else 0
    results.add_test("Overall", passed=overall >= 0.8, score=overall, threshold=0.8)
    
    if overall >= 0.8:
        print("✓ Test 2 PASSED")
    else:
        print("✗ Test 2 FAILED")
    
    # Save results
    results.save()
    print(f"\nResults saved")
    
    return 0 if overall >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
