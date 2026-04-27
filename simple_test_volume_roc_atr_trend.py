#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test if strategy can be imported from registry
try:
    from crabquant.strategies import STRATEGY_REGISTRY
    
    # Check if our new strategy is in registry
    if 'invented_volume_roc_atr_trend' in STRATEGY_REGISTRY:
        print("✓ invented_volume_roc_atr_trend successfully registered in STRATEGY_REGISTRY")
        
        # Get strategy components
        signals_fn, defaults, grid, desc, matrix_fn = STRATEGY_REGISTRY['invented_volume_roc_atr_trend']
        
        print(f"✓ DEFAULT_PARAMS: {defaults}")
        print(f"✓ PARAM_GRID has {len(grid)} parameters")
        print(f"✓ DESCRIPTION length: {len(desc)} characters")
        
        # Test that functions are callable
        if callable(signals_fn) and callable(matrix_fn):
            print("✓ Both generate_signals and generate_signals_matrix are callable")
        else:
            print("✗ Functions are not callable")
            
    else:
        print("✗ invented_volume_roc_atr_trend not found in STRATEGY_REGISTRY")
        
    # Test direct import
    try:
        from crabquant.strategies.invented_volume_roc_atr_trend import generate_signals, DEFAULT_PARAMS
        print("✓ Direct import successful")
        print(f"✓ Direct DEFAULT_PARAMS: {DEFAULT_PARAMS}")
    except Exception as e:
        print(f"✗ Direct import failed: {e}")
        
except Exception as e:
    print(f"✗ Registry test failed: {e}")

print("\nStrategy validation complete.")