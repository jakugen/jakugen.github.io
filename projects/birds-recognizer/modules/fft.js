'use strict';

/**
 * Compute the Fast Fourier Transform (FFT) of a real-valued signal.
 * The input is a real-valued array. The output is an object with two arrays:
 *   { real: [...], imag: [...] }
 * Note: The length of the input must be a power of 2.
 */
export function myFft(signal) {
  const N = signal.length;
  if (N === 0) return { real: [], imag: [] };
  if (N === 1) return { real: [signal[0]], imag: [0] };

  // Check if N is a power of 2
  if ((N & (N - 1)) !== 0) {
    throw new Error("Signal length is not a power of 2.");
  }

  // Separate even and odd indexed elements
  const evenSignal = [];
  const oddSignal = [];
  for (let i = 0; i < N; i++) {
    (i % 2 === 0 ? evenSignal : oddSignal).push(signal[i]);
  }

  // Recursively compute FFT for even and odd parts
  const evenFFT = myFft(evenSignal);
  const oddFFT = myFft(oddSignal);

  // Initialize output arrays
  const real = new Array(N).fill(0);
  const imag = new Array(N).fill(0);

  // Combine
  for (let k = 0; k < N / 2; k++) {
    const angle = (-2 * Math.PI * k) / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    // Multiply oddFFT[k] by twiddle factor e^(-j*2pi*k/N)
    const oddReal = oddFFT.real[k] * cos - oddFFT.imag[k] * sin;
    const oddImag = oddFFT.real[k] * sin + oddFFT.imag[k] * cos;

    real[k] = evenFFT.real[k] + oddReal;
    imag[k] = evenFFT.imag[k] + oddImag;

    real[k + N / 2] = evenFFT.real[k] - oddReal;
    imag[k + N / 2] = evenFFT.imag[k] - oddImag;
  }

  return { real, imag };
}

/**
 * FFT stub that calls the above myFft.
 */
export function fft(inputArray) {
  // We assume inputArray is a real-valued array.
  return myFft(inputArray);
}