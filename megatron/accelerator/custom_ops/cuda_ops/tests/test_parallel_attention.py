import argparse

from megatron.accelerator.custom_ops.cuda_ops import load

def test_parallel_attention():
    import parallel_attention

    if parallel_attention.update():
        print("[Success] test_parallel_attention")
    else:
        print("[Fail] test_parallel_attention")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.rank = 0
    args.masked_softmax_fusion = False

    load(args)
    test_parallel_attention()
