import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "libero"))

from models import device as dev  # noqa: E402
from models.transformer_smolvlm import SmolVLMActionTransformer  # noqa: E402


def fake_host(xpu=False, cuda=False):
    return mock.patch.object(dev, "is_available", lambda name: {"xpu": xpu, "cuda": cuda}.get(name, name == "cpu"))


class DetectionTest(unittest.TestCase):
    def test_xpu_beats_cuda(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.resolve_device(None), torch.device("xpu", 0))

    def test_cuda_when_no_xpu(self):
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda", 0))

    def test_cpu_fallback(self):
        with fake_host():
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("xpu"), torch.device("xpu", 0))


class ServerDispatchTest(unittest.TestCase):
    def setUp(self):
        import serve_smolvlm_libero as serve

        self.serve = serve

    def _load(self, device_name):
        fake_model = mock.MagicMock()
        fake_model.to.return_value = fake_model
        with mock.patch.object(self.serve.SmolVLMVLA, "from_pretrained", return_value=fake_model), \
                mock.patch.object(self.serve.SmolVLMVLAProcessor, "from_pretrained"):
            self.serve.load_model("ckpt", device_name=device_name)
        return fake_model

    def test_parser_device_defaults_to_detect(self):
        parser = self.serve.build_parser()
        self.assertIsNone(parser.parse_args(["--checkpoint", "c"]).device)
        self.assertEqual(parser.parse_args(["--checkpoint", "c", "--device", "cpu"]).device, "cpu")

    def test_model_moved_to_detected_device(self):
        with fake_host(xpu=True):
            model = self._load(None)
        model.to.assert_called_once_with(torch.device("xpu", 0))

    def test_model_moved_to_cuda_host_device(self):
        with fake_host(cuda=True):
            model = self._load(None)
        model.to.assert_called_once_with(torch.device("cuda", 0))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            model = self._load("cpu")
        model.to.assert_called_once_with(torch.device("cpu"))


@unittest.skipUnless(dev.is_available("xpu"), "needs an Intel XPU")
class XpuParityTest(unittest.TestCase):
    def test_action_transformer_matches_cpu(self):
        torch.manual_seed(0)
        model = SmolVLMActionTransformer(
            hidden_size=64, vlm_hidden_size=32, depth=2, num_heads=4,
            dim_action=7, dim_propio=8, max_len_seq=64,
        ).eval()
        for p in model.parameters():  # re-randomise zero-initialised layers
            torch.nn.init.normal_(p, std=0.05)
        vlm = torch.randn(1, 12, 32)
        x = torch.randn(1, 10, 7)
        proprio = torch.randn(1, 8)
        t = torch.tensor([0.5])
        with torch.no_grad():
            ref = model(vlm_features=vlm, action_with_noise=x, proprio=proprio, t=t)
            xpu = torch.device("xpu", 0)
            out = model.to(xpu)(
                vlm_features=vlm.to(xpu), action_with_noise=x.to(xpu), proprio=proprio.to(xpu), t=t.to(xpu)
            ).cpu()
        self.assertGreater(ref.std().item(), 1e-4)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
