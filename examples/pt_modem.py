from rfml.ptradio import AWGN, Transmitter, Receiver, theoreticalBER

import numpy as np

modulation = "BPSK"  # could be QPSK, 8PSK, QAM16, QAM64
tx = Transmitter(modulation=modulation)
channel = AWGN()
rx = Receiver(modulation=modulation)

n_symbols = int(10e3)
n_bits = int(tx.symbol_encoder.get_bps() * n_symbols)
snrs = list(range(0, 8))
n_trials = 10

for snr in range(0, 8):
    channel.set_snr(snr)
    n_errors = 0

    for _ in range(n_trials):
        tx_bits = np.random.randint(low=0, high=2, size=n_bits)
        tx_iq = tx.modulate(bits=tx_bits)

        rx_iq = channel(tx_iq)

        rx_bits = rx.demodulate(iq=rx_iq)
        rx_bits = np.array(rx_bits)

        n_errors += np.sum(np.abs(tx_bits - rx_bits))

    ber = float(n_errors) / float(n_bits * n_trials)
    theory = theoreticalBER(modulation=modulation, snr=snr)

    print(
        "BER={:.3e}, "
        "theory={:.3e}, "
        "|diff|={:.3e}, "
        "SNR={:d}, "
        "modulation={}".format(ber, theory, np.abs(ber - theory), snr, modulation)
    )
