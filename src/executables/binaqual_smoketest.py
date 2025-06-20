

if __name__ == "__main__":
    import torch, sys
    from binaqual import calculate_binaqual  # uses your patched loader

    # --------------------------------------------------------------------
    # 1. create a random stereo tensor  (channels, samples) = (2, 32 000)
    tensor = torch.rand(2, 32_000, dtype=torch.float32)  # 2-s clip @16 kHz

    # --------------------------------------------------------------------
    # 2. run Binaqual directly on the tensor (ref == test)
    nsim, ls = calculate_binaqual(tensor, tensor)
    print(f"NSIM: {nsim}")  # expect [1.0, 1.0]
    print(f"LS  : {ls}")  # expect 1.0  (= product of the two NSIMs)

    # --------------------------------------------------------------------
    # 3. sanity check
    if not (all(abs(n - 1.0) < 1e-6 for n in nsim) and abs(ls - 1.0) < 1e-6):
        sys.exit("❌  tensor smoke-test failed")
    print("✅  tensor smoke-test passed")