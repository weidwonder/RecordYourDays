import unittest

from dairy_engine.gpu import parse_nvidia_smi, pick_gpus


class GpuTest(unittest.TestCase):
    def test_pick(self):
        gpus = parse_nvidia_smi("0, GPU-a, 16, 22528\n1, GPU-b, 6000, 22528\n", "GPU-b, 4242, 5990\n")
        self.assertEqual(pick_gpus(gpus, 10000), [0], "有其他进程在用的卡不用")
        gpus = parse_nvidia_smi("0, GPU-a, 16000, 22528\n1, GPU-b, 6, 22528\n", "")
        self.assertEqual(pick_gpus(gpus, 10000), [1], "剩余显存不足的卡不用")
        self.assertEqual(pick_gpus(gpus, 1000, allowed={0}), [0])
        self.assertEqual(pick_gpus(parse_nvidia_smi("0, GPU-a, 0, 22528", ""), 10000), [0])


if __name__ == "__main__":
    unittest.main()
