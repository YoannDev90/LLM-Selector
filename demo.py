#!/usr/bin/env python3
"""
Quick demo script to test LLM Selector basic functionality
"""

from llm_selector import LLMSelector, LLMSelectorConfig

def main():
    print("🤖 LLM Selector Demo")
    print("=" * 50)

    # Test LLMSelectorConfig
    print("\n1. Testing LLMSelectorConfig...")
    config = LLMSelectorConfig("gpt-4", api_key="demo-key")
    print(f"✓ Config created: {config.model}")

    validation = config.validate()
    print(f"✓ Config validation: {'PASS' if validation else 'FAIL'}")

    # Test LLMSelector instantiation
    print("\n2. Testing LLMSelector instantiation...")
    models = {
        "gpt-4": {"description": "GPT-4 for complex tasks", "weight": 2.0},
        "claude-3": {"description": "Claude 3 for balanced tasks", "weight": 1.0}
    }

    configs = [LLMSelectorConfig("gpt-4", api_key="demo-key")]

    try:
        selector = LLMSelector(models, selector_configs=configs, debug=True)
        print("✓ LLMSelector created successfully")
        print(f"✓ Debug mode: {selector.debug}")
        print(f"✓ Max cache size: {selector.max_cache_size}")
    except Exception as e:
        print(f"✗ Error creating LLMSelector: {e}")
        return

    # Test parse_selection (if available)
    print("\n3. Testing parse_selection...")
    try:
        result = selector.parse_selection("I recommend gpt-4")
        print(f"✓ Parse selection result: {result}")
    except AttributeError:
        print("⚠ parse_selection method not fully implemented yet")
    except Exception as e:
        print(f"✗ Error in parse_selection: {e}")

    print("\n4. Demo completed!")
    print("\nNote: Full functionality requires completing the LLMSelector implementation")
    print("See TODO_IMPLEMENTATION.md for details on missing features.")

if __name__ == "__main__":
    main()