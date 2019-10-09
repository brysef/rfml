# External Includes
import numpy as np
import torch
from typing import List

# Internal Includes
from .constellation import ConstellationMapper, ConstellationUnmapper
from .downsample import Downsample
from .rrc import RRC
from .upsample import Upsample


def theoreticalBER(snr: int, modulation: str) -> float:
    """Lookup the theoretical BER for a given modulation scheme that has been
    precomputed using MATLAB's *berawgn*.

    Args:
        snr (int): Signal-to-Noise ratio (:math:`E_s/N_0` not :math:`E_b/N_0`).
        modulation (str): Name of the modulation format -- currently supported options
                          are:
                            - BSPK
                            - QPSK
                            - 8PSK
                            - QAM16
                            - QAM64

    Raises:
        ValueError: If snr is not in [0-20].
        ValueError: If modulation is unknown

    Returns:
        float: Theoretical BER @ snr (Es/N0) for modulation

    .. note::

        MatLab takes in :math:`E_b/N_0` for calculations as that is what is
        typically used and plotted in most literature.  This code has chosen to use
        :math:`E_s/N_0` instead which is easily related back to :math:`E_b/N_0` through:

        .. math::

            \\frac{E_s}{N_0} = \\frac{E_b}{N_0} - 10*log10(log2(M))

        Where :math:`M` is the order of the modulation.
    """
    if snr < 0 or snr > 20:
        raise ValueError("SNR must be in the range of 0-20 ({})".format(snr))

    modulation = modulation.upper()

    if modulation == "BPSK":
        # berawgn((0:20), 'psk', 2, 'nondiff')
        bpskBER = [
            0.07865,
            0.056282,
            0.037506,
            0.022878,
            0.012501,
            0.0059539,
            0.0023883,
            0.00077267,
            0.00019091,
            3.3627e-05,
            3.8721e-06,
            2.6131e-07,
            9.006e-09,
            1.3329e-10,
            6.8102e-13,
            9.124e-16,
            2.2674e-19,
            6.759e-24,
            1.396e-29,
            1.0011e-36,
            1.0442e-45,
        ]
        return bpskBER[snr]
    elif modulation == "QPSK":
        # berawgn((0:20) - 10*log10(log2(4)), 'psk', 4, 'nondiff')
        qpskBER = [
            0.15866,
            0.13093,
            0.10403,
            0.078896,
            0.056495,
            0.037679,
            0.023007,
            0.012587,
            0.0060044,
            0.0024133,
            0.0007827,
            0.00019399,
            3.4303e-05,
            3.9692e-06,
            2.6951e-07,
            9.361e-09,
            1.399e-10,
            7.236e-13,
            9.845e-16,
            2.4945e-19,
            7.6199e-24,
        ]
        return qpskBER[snr]
    elif modulation == "8PSK":
        # berawgn((0:20) - 10*log10(log2(8)), 'psk', 8, 'nondiff')
        psk8BER = [
            0.24115,
            0.21586,
            0.19029,
            0.16504,
            0.14064,
            0.11754,
            0.096025,
            0.076242,
            0.058318,
            0.042467,
            0.029013,
            0.018277,
            0.010399,
            0.0052101,
            0.0022266,
            0.00077982,
            0.00021283,
            4.2476e-05,
            5.7223e-06,
            4.704e-07,
            2.0779e-08,
        ]
        return psk8BER[snr]
    elif modulation == "QAM16":
        # berawgn((0:20) - 10*log10(log2(16)), 'qam', 16)
        qam16BER = [
            0.28728,
            0.26248,
            0.23723,
            0.21216,
            0.18774,
            0.16417,
            0.14144,
            0.11944,
            0.098171,
            0.077858,
            0.058993,
            0.042212,
            0.02813,
            0.017159,
            0.0093756,
            0.0044654,
            0.0017912,
            0.00057951,
            0.00014318,
            2.522e-05,
            2.9041e-06,
        ]
        return qam16BER[snr]
    elif modulation == "QAM64":
        # berawgn((0:20) - 10*log10(log2(64)), 'qam', 64)
        qam64BER = [
            0.35986,
            0.34279,
            0.32447,
            0.30492,
            0.28421,
            0.26251,
            0.2401,
            0.21743,
            0.19498,
            0.17324,
            0.15255,
            0.13302,
            0.11458,
            0.097022,
            0.080203,
            0.064159,
            0.049171,
            0.035695,
            0.024217,
            0.015106,
            0.0084864,
        ]
        return qam64BER[snr]

    raise ValueError("Unknown modulation scheme: {}".format(modulation))


class Transmitter(object):
    """Class containing a full transmit chain.

    The basic chain structure can be described as:

    ::

            [Random]━┓
        [User Input]━┻━[Unpack]━━[Constellation Mapper]━━[Upsample]━━[RRC]━━[Output]

    After being constructed, the user can call modulate either with their own data
    (shown as "User Input" above) or they can pass in a set number of symbols in order
    to have "Random" data generated.  Either way, the output is always IQ at baseband.

    Args:
        modulation (str): Linear modulation format to use.  Currently supported values
                          are:
                            - BPSK
                            - QPSK
                            - 8PSK
                            - QAM16
                            - QAM64
        alpha (float, optional): Roll-off factor for the RRC filter. Defaults to 0.35.
        sps (int, optional): Sample per symbol for Upsample. Defaults to 8.
        filter_span (int, optional): Half-sided RRC filter span in symbols. Defaults to
                                     8.

    Raises:
        ValueError: If the constellation is unknown.
        ValueError: If sps is not at least 2.
        ValueError: If alpha is not in (0, 1).
        ValueError: If filter span is not positive.
    """

    def __init__(
        self, modulation: str, alpha: float = 0.35, sps: int = 8, filter_span: int = 8
    ):
        constellation = _get_constellation(modulation=modulation)
        self.symbol_encoder = ConstellationMapper(constellation=constellation)
        self.upsample = Upsample(i=sps)
        self.pulse_shape = RRC(
            alpha=alpha, sps=sps, filter_span=filter_span, add_pad=True
        )

    def modulate(self, bits: List[int] = None, n_symbols: int = 10000) -> torch.Tensor:
        """Modulate a provided list of bits (1s and 0s) or random data of a set length.

        If you wish to provide your own data, which is useful for later calculating the
        bit error rate, then you can directly pass in a list of bits that have been
        generated by your application (or randomly generated).

        If instead you wish to simply get some quick data examples and don't care about
        the underlying bit stream, then you can call this function with no arguments or
        override n_symbols to generate longer/shorter sequences.

        Args:
            bits (List[int], optional): List of bits to modulate. Defaults to None.
            n_symbols (int, optional): Number of random symbols to generate. Defaults to
                                       10000.

        Raises:
            ValueError: If bits is not provided and you set the number of symbols <= 0.
            ValueError: If the bit stream would have to be zero padded for transmisison.
            ValueError: If *bits* contains values other than just 1 and 0.

        Returns:
            torch.Tensor: Modulated data at baseband (1x1x2xn_symbols)
        """
        if bits is None and n_symbols <= 0:
            raise ValueError(
                "Must either provide data or a positive number of symbols to be faked "
                "-- you provided {}".format(n_symbols)
            )
        if bits is not None and (len(bits) % self.symbol_encoder.get_bps()) != 0:
            raise ValueError(
                "The number of bits input ({n_bits:d}) would have to be padded to "
                "be transmitted because the bits per symbol in the constellation "
                "is {bps:d}.  It is highly unlikely that was your intention and "
                "therefore we're complaining now to avoid errors in your code "
                "later.".format(n_bits=len(bits), bps=self.symbol_encoder.get_bps())
            )

        # Encode from "bits" to "chunks" which are simply indexes of the symbols
        # that are expected by the symbol encoder as a LongTensor (int64)
        if bits is not None:
            chunks = list()
            bin_str = ""
            for i, bit in enumerate(bits):
                # Force a binary encoding here by checking for 1, 0 equality
                # If the value was not 1,0 then it won't match back up coming out of
                # the receiver anyways and would therefore cause an error
                if bit == 0:
                    bin_str += "0"
                elif bit == 1:
                    bin_str += "1"
                else:
                    raise ValueError(
                        "Value of the passed in data at loc {} is not binary (e.g. "
                        "1 or 0).  The value is {}".format(i, bit)
                    )
                if len(bin_str) == self.symbol_encoder.get_bps():
                    chunks.append(int(bin_str, base=2))
                    bin_str = ""
            chunks = np.array(chunks)
        else:
            chunks = np.random.randint(
                low=0, high=self.symbol_encoder.get_M(), size=n_symbols
            )
        chunks = torch.from_numpy(chunks.astype(np.int64))

        # Modulate the baseband signal in complex format (represented as BxCxIQxN)
        symbols = self.symbol_encoder(chunks)
        signal = self.upsample(symbols)
        ret = self.pulse_shape(signal)

        return ret


class Receiver(object):
    """Class containing a full receive chain.

    The basic chain structure can be described as:

    ::

        [Baseband IQ Input]━━[RRC]━━[Downsample]━━[Constellation Unmapper]━━[Pack]━━[Output]

    After being constructed, the user can call demodulate either with their own IQ data
    (shown as "Baseband IQ Input" above) and this data is assumed to be formatted as
    (1x1x2xN) which is the output of the Transmitter.  The output of demodulate is then
    a list of bits (1s and 0s) that can be analyzed for bit error rate calculations or
    other applications.

    Args:
        modulation (str): Modulation format to use.  Currently supported values are:
                            - BPSK
                            - QPSK
                            - 8PSK
                            - QAM16
                            - QAM64
        alpha (float, optional): Roll-off factor for the RRC filter. Defaults to 0.35.
        sps (int, optional): Sample per symbol for Upsample. Defaults to 8.
        filter_span (int, optional): Half-sided RRC filter span in symbols. Defaults to
                                     8.

    Raises:
        ValueError: If the constellation is unknown.
        ValueError: If sps is not at least 2.
        ValueError: If alpha is not in (0, 1).
        ValueError: If filter span is not positive.

    .. warning::

        This receive chain is very simplistic and assumes both frame and symbol
        synchronization -- in other words, its meant as a simple simulation toy and not
        as an actual receiver implementation.
    """

    def __init__(
        self, modulation: str, alpha: float = 0.35, sps: int = 8, filter_span: int = 8
    ):
        constellation = _get_constellation(modulation=modulation)
        self.match_filter = RRC(
            alpha=alpha, sps=sps, filter_span=filter_span, add_pad=False
        )
        self.downsample = Downsample(offset=sps * filter_span, d=sps)
        self.hard_decision = ConstellationUnmapper(constellation=constellation)

    def demodulate(self, iq: torch.Tensor) -> List[int]:
        """Demodulate a signal at baseband and return a list of bits (1s and 0s).

        Args:
            iq (torch.Tensor): Complex baseband signal in (1x1x2xN) as output by
                               Transmitter.modulate().

        Raises:
            ValueError: If the provided IQ does not have the shape (1x1x2xN).

        Returns:
            List[int]: Demodulated bits.
        """
        if (
            len(iq.shape) != 4
            or iq.shape[0] != 1
            or iq.shape[1] != 1
            or iq.shape[2] != 2
        ):
            raise ValueError(
                "The provided IQ Tensor must have the shape 1x1x2xN -- "
                "but your's has shape {}".format(iq.shape)
            )

        # Demodulate the baseband signal assuming synchronization
        filtered = self.match_filter(iq)
        symbol_estimates = self.downsample(filtered)
        chunks = self.hard_decision(symbol_estimates)

        # Convert the "chunks" back into an binary array (1s and 0s)
        ret = list()
        for i in range(chunks.shape[3]):
            bin_str = bin(chunks[0, 0, 0, i])
            # bin_str -> 0b0101 therefore ignore the first two characters
            bin_str = bin_str[2:]
            # Zero pad the beginning of the string
            while len(bin_str) < self.hard_decision.get_bps():
                bin_str = "0" + bin_str

            # Encode the rest of them as integers in the list and add some defensive
            # programming in case I've screwed up the logic of this conversion
            for bin_char in bin_str:
                if bin_char == "0":
                    ret.append(0)
                elif bin_char == "1":
                    ret.append(1)
                else:
                    raise RuntimeError(
                        "Unknown value encounter from bin() -- {}".format(bin_char)
                    )

        return ret


def _get_constellation(modulation: str) -> np.ndarray:
    modulation = modulation.upper()
    if modulation == "BPSK":
        return _normalize_and_convert(_bpsk_constellation)
    elif modulation == "QPSK":
        return _normalize_and_convert(_qpsk_constellation)
    elif modulation == "8PSK":
        return _normalize_and_convert(_8psk_constellation)
    elif modulation == "QAM16":
        return _normalize_and_convert(_qam16_constellation)
    elif modulation == "QAM64":
        return _normalize_and_convert(_qam64_constellation)
    else:
        raise ValueError("Unknown Modulation Scheme ({})".format(modulation))


def _normalize_and_convert(constellation: List[float]) -> np.ndarray:
    """Normalize constellation power (Es) to 1 and convert to real numbers (2xM)
    """
    power = np.mean(np.abs(np.array(constellation)) ** 2)
    constellation = constellation / np.sqrt(power)
    ret = np.stack((constellation.real, constellation.imag))
    return ret


# Private Constants Definitions for Grey Coded Constellation Maps
_bpsk_constellation = ((1 + 0j), (-1 + 0j))
_qpsk_constellation = (
    (0.7071067690849304 + 0.7071067690849304j),
    (-0.7071067690849304 + 0.7071067690849304j),
    (0.7071067690849304 - 0.7071067690849304j),
    (-0.7071067690849304 - 0.7071067690849304j),
)
_8psk_constellation = (
    (1 + 0j),
    (0.7071067690849304 + 0.7071067690849304j),
    (-0.7071067690849304 + 0.7071067690849304j),
    (0 + 1j),
    (0.7071067690849304 - 0.7071067690849304j),
    (0 - 1j),
    (-1 + 0j),
    (-0.7071067690849304 - 0.7071067690849304j),
)
_qam16_constellation = (
    (-1.001551628112793 - 1.001551628112793j),
    (-1.001551628112793 - 0.33385056257247925j),
    (-1.001551628112793 + 1.001551628112793j),
    (-1.001551628112793 + 0.33385056257247925j),
    (-0.33385056257247925 - 1.001551628112793j),
    (-0.33385056257247925 - 0.33385056257247925j),
    (-0.33385056257247925 + 1.001551628112793j),
    (-0.33385056257247925 + 0.33385056257247925j),
    (1.001551628112793 - 1.001551628112793j),
    (1.001551628112793 - 0.33385056257247925j),
    (1.001551628112793 + 1.001551628112793j),
    (1.001551628112793 + 0.33385056257247925j),
    (0.33385056257247925 - 1.001551628112793j),
    (0.33385056257247925 - 0.33385056257247925j),
    (0.33385056257247925 + 1.001551628112793j),
    (0.33385056257247925 + 0.33385056257247925j),
)
_qam64_constellation = (
    (-1.1500126123428345 - 1.1500126123428345j),
    (-1.1500126123428345 - 0.8214375972747803j),
    (-1.1500126123428345 - 0.1642875224351883j),
    (-1.1500126123428345 - 0.4928625524044037j),
    (-1.1500126123428345 + 1.1500126123428345j),
    (-1.1500126123428345 + 0.8214375972747803j),
    (-1.1500126123428345 + 0.1642875224351883j),
    (-1.1500126123428345 + 0.4928625524044037j),
    (-0.8214375972747803 - 1.1500126123428345j),
    (-0.8214375972747803 - 0.8214375972747803j),
    (-0.8214375972747803 - 0.1642875224351883j),
    (-0.8214375972747803 - 0.4928625524044037j),
    (-0.8214375972747803 + 1.1500126123428345j),
    (-0.8214375972747803 + 0.8214375972747803j),
    (-0.8214375972747803 + 0.1642875224351883j),
    (-0.8214375972747803 + 0.4928625524044037j),
    (-0.1642875224351883 - 1.1500126123428345j),
    (-0.1642875224351883 - 0.8214375972747803j),
    (-0.1642875224351883 - 0.1642875224351883j),
    (-0.1642875224351883 - 0.4928625524044037j),
    (-0.1642875224351883 + 1.1500126123428345j),
    (-0.1642875224351883 + 0.8214375972747803j),
    (-0.1642875224351883 + 0.1642875224351883j),
    (-0.1642875224351883 + 0.4928625524044037j),
    (-0.4928625524044037 - 1.1500126123428345j),
    (-0.4928625524044037 - 0.8214375972747803j),
    (-0.4928625524044037 - 0.1642875224351883j),
    (-0.4928625524044037 - 0.4928625524044037j),
    (-0.4928625524044037 + 1.1500126123428345j),
    (-0.4928625524044037 + 0.8214375972747803j),
    (-0.4928625524044037 + 0.1642875224351883j),
    (-0.4928625524044037 + 0.4928625524044037j),
    (1.1500126123428345 - 1.1500126123428345j),
    (1.1500126123428345 - 0.8214375972747803j),
    (1.1500126123428345 - 0.1642875224351883j),
    (1.1500126123428345 - 0.4928625524044037j),
    (1.1500126123428345 + 1.1500126123428345j),
    (1.1500126123428345 + 0.8214375972747803j),
    (1.1500126123428345 + 0.1642875224351883j),
    (1.1500126123428345 + 0.4928625524044037j),
    (0.8214375972747803 - 1.1500126123428345j),
    (0.8214375972747803 - 0.8214375972747803j),
    (0.8214375972747803 - 0.1642875224351883j),
    (0.8214375972747803 - 0.4928625524044037j),
    (0.8214375972747803 + 1.1500126123428345j),
    (0.8214375972747803 + 0.8214375972747803j),
    (0.8214375972747803 + 0.1642875224351883j),
    (0.8214375972747803 + 0.4928625524044037j),
    (0.1642875224351883 - 1.1500126123428345j),
    (0.1642875224351883 - 0.8214375972747803j),
    (0.1642875224351883 - 0.1642875224351883j),
    (0.1642875224351883 - 0.4928625524044037j),
    (0.1642875224351883 + 1.1500126123428345j),
    (0.1642875224351883 + 0.8214375972747803j),
    (0.1642875224351883 + 0.1642875224351883j),
    (0.1642875224351883 + 0.4928625524044037j),
    (0.4928625524044037 - 1.1500126123428345j),
    (0.4928625524044037 - 0.8214375972747803j),
    (0.4928625524044037 - 0.1642875224351883j),
    (0.4928625524044037 - 0.4928625524044037j),
    (0.4928625524044037 + 1.1500126123428345j),
    (0.4928625524044037 + 0.8214375972747803j),
    (0.4928625524044037 + 0.1642875224351883j),
    (0.4928625524044037 + 0.4928625524044037j),
)
