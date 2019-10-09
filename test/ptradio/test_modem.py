"""Essentially an integration test by checking that BER curves match theory in AWGN.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
from rfml.ptradio import AWGN, Transmitter, Receiver, theoreticalBER


class TestModem(TestCase):
    def test_BPSK(self):
        self._test_es_is_1(modulation="BPSK")
        self._test_awgn(modulation="BPSK")

    def test_QPSK(self):
        self._test_es_is_1(modulation="QPSK")
        self._test_awgn(modulation="QPSK")

    def test_8PSK(self):
        self._test_es_is_1(modulation="8PSK")
        self._test_awgn(modulation="8PSK")

    def test_QAM16(self):
        self._test_es_is_1(modulation="QAM16")
        self._test_awgn(modulation="QAM16")

    def test_QAM64(self):
        self._test_es_is_1(modulation="QAM64")
        self._test_awgn(modulation="QAM64")

    def _test_es_is_1(self, modulation: str):
        sps = 8
        tx = Transmitter(modulation=modulation, sps=sps)
        iq = tx.modulate()

        power = _power(iq)

        self.assertAlmostEqual(power, 1.0 / float(sps), delta=0.005)

    def _test_awgn(self, modulation: str):
        sps = 8
        tx = Transmitter(modulation=modulation, sps=sps)
        channel = AWGN()
        rx = Receiver(modulation=modulation, sps=sps)

        n_symbols = int(10e3)
        n_bits = int(tx.symbol_encoder.get_bps() * n_symbols)
        snrs = list(range(0, 8))
        n_trials = 10

        for snr in snrs:
            channel.set_snr(snr)
            n_errors = 0
            for _ in range(n_trials):
                tx_bits = np.random.randint(low=0, high=2, size=n_bits)
                tx_iq = tx.modulate(bits=tx_bits)

                rx_iq = channel(tx_iq)

                # Verify that the SNR is correct thereby validating AWGN and eliminating
                # a possible source of error in the BER
                noise = rx_iq - tx_iq
                snr_estimate = 10.0 * np.log10(_power(tx_iq) * sps / _power(noise))
                print("SNR={:.2f}, SNR_Estimate={:.2f}".format(snr, snr_estimate))
                self.assertLessEqual(np.abs(snr - snr_estimate), 0.2)

                rx_bits = rx.demodulate(iq=rx_iq)
                rx_bits = np.array(rx_bits)

                self.assertEqual(len(rx_bits.shape), 1)
                self.assertEqual(rx_bits.shape[0], tx_bits.shape[0])

                n_errors += np.sum(np.abs(tx_bits - rx_bits))
            ber = float(n_errors) / float(n_bits * n_trials)
            theory = theoreticalBER(modulation=modulation, snr=snr)

            print(
                "BER={:.3f}, "
                "theory={:.3f}, "
                "|diff|={:.3f}, "
                "SNR={:d}, "
                "modulation={}".format(
                    ber, theory, np.abs(ber - theory), snr, modulation
                )
            )

            self.assertLessEqual(np.abs(ber - theory), 50e-4)


def _power(x: torch.Tensor) -> float:
    # Write our own power function as to not depend on rfml.nn.F.energy
    x = x.numpy()[0, 0, :]
    s = x[0, :] + x[1, :] * 1j

    power = np.mean(np.abs(s) ** 2)

    return power
