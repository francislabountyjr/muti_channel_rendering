import unittest

import numpy as np

from binaural_channel import EXPECTED_714_ORDER, binaural_convolve, build_hrtf_filters


class BinauralOrientationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hrtf_left, cls.hrtf_right = build_hrtf_filters("HRIR_L2702.sofa", verbose=False)

    def render_impulse(self, speaker_name):
        speaker_index = EXPECTED_714_ORDER.index(speaker_name)
        audio = np.zeros((1, len(EXPECTED_714_ORDER)), dtype=np.float32)
        audio[0, speaker_index] = 1.0
        return binaural_convolve(audio, self.hrtf_left, self.hrtf_right)

    def assert_dominant_ear(self, speaker_name, expected_ear):
        output = self.render_impulse(speaker_name)
        left_energy = float(np.sum(np.abs(output[:, 0])))
        right_energy = float(np.sum(np.abs(output[:, 1])))

        if expected_ear == "left":
            self.assertGreater(left_energy, right_energy, speaker_name)
        else:
            self.assertGreater(right_energy, left_energy, speaker_name)

    def test_left_speakers_favor_left_ear(self):
        for speaker_name in ("FL", "BL", "SL", "TFL", "TBL"):
            with self.subTest(speaker=speaker_name):
                self.assert_dominant_ear(speaker_name, "left")

    def test_right_speakers_favor_right_ear(self):
        for speaker_name in ("FR", "BR", "SR", "TFR", "TBR"):
            with self.subTest(speaker=speaker_name):
                self.assert_dominant_ear(speaker_name, "right")


if __name__ == "__main__":
    unittest.main()
