print("Stable Diffusion 1.4 test is running!")
print("Loading test file...")

def test_simple():
    """Simple test to verify pytest is working"""
    print("Simple test is running!")
    assert True
    print("Simple test passed!")

def test_basic():
    """Basic test without complex imports"""
    print("Basic test is running!")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    assert True
    print("Basic test passed!") 