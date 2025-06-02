package com.neocoretechs.wordembedding;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;


/**
 * Implementation of Super-Bit Locality-Sensitive Hashing.
 * Super-Bit is an improvement of Random Projection LSH.
 * It computes an estimation of cosine similarity.
 *
 * Super-Bit Locality-Sensitive Hashing
 * Jianqiu Ji, Jianmin Li, Shuicheng Yan, Bo Zhang, Qi Tian
 * http://papers.nips.cc/paper/4847-super-bit-locality-sensitive-hashing.pdf
 * Advances in Neural Information Processing Systems 25, 2012
 *
 * Supported input types:
 * - double[]
 * @author original:Thibault Debatty
 * @author Groff
 */
final class SuperBit implements java.io.Serializable, Comparable {
	private static final long serialVersionUID = -1L;
	private boolean[] sig;
    private transient double[][] hyperplanes;
    private static final int DEFAULT_CODE_LENGTH = 10000;
    /**
     * Initialize SuperBit algorithm.
     * Super-Bit depth n must be [1 .. d] and number of Super-Bit l in [1 ..
     * The resulting code length k = n * l
     * The K vectors are orthogonalized in L batches of N vectors
     *
     * @param d data space dimension
     * @param n Super-Bit depth [1 .. d]
     * @param l number of Super-Bit [1 ..
     */
    public SuperBit(final int d, final int n, final int l) {
        this(d, n, l, new Random());
    }
    /**
     * Initialize SuperBit algorithm.
     * Super-Bit depth n must be [1 .. d] and number of Super-Bit l in [1 ..
     * The resulting code length k = n * l
     * The K vectors are orthogonalized in L batches of N vectors
     *
     * @param d data space dimension
     * @param n Super-Bit depth [1 .. d]
     * @param l number of Super-Bit [1 ..
     * @param seed to use for the random number generator
     */
    public SuperBit(final int d, final int n, final int l, final long seed) {
        this(d, n, l, new Random(seed));
    }
    private SuperBit(final int d, final int n, final int l, final Random rand) {
        if (d <= 0) {
            throw new IllegalArgumentException("Dimension d must be >= 1");
        }
        if (n < 1 || n > d) {
            throw new IllegalArgumentException(
                    "Super-Bit depth N must be 1 <= N <= d");
        }
        if (l < 1) {
            throw  new IllegalArgumentException(
                    "Number of Super-Bit L must be >= 1");
        }
        // Input: Data space dimension d, Super-Bit depth 1 <= N <= d,
        // number of Super-Bit L >= 1,
        // resulting code length K = N * L
        // Generate a random matrix H with each element sampled independently
        // from the normal distribution
        // N (0, 1), with each column normalized to unit length.
        // Denote H = [v1, v2, ..., vK].
        int code_length = n * l;
        double[][] v = new double[code_length][d];
        Parallel.parallelFor(0, code_length, t -> {
            double[] vector = new double[d];
            for (int j = 0; j < d; j++) {
                vector[j] = rand.nextGaussian();
            }
            normalize(vector);
            v[t] = vector;
        });
        double[][] w = new double[code_length][d];
        for (int i = 0; i <= l - 1; i++) {
            for (int j = 1; j <= n; j++) {
                java.lang.System.arraycopy(
                        v[i * n + j - 1],
                        0,
                        w[i * n + j - 1],
                        0,
                        d);

                for (int k = 1; k <= (j - 1); k++) {
                    w[i * n + j - 1] = sub(
                            w[i * n + j - 1],
                            product(
                                    dotProduct(
                                            w[i * n + k - 1],
                                            v[ i * n + j - 1]),
                                    w[i * n + k - 1]));
                }
                normalize(w[i * n + j - 1]);
            }
        }
        this.hyperplanes = w;
    }
    /**
     * Initialize SuperBit algorithm.
     * With code length K = 10000
     * The K vectors are orthogonalized in d batches of 10000/d vectors
     * The resulting mean error is 0.01
     * @param d The size of the vector we are operating on
     */
    public SuperBit(final int d) {
        this(d, d, DEFAULT_CODE_LENGTH / d, 8675309);
    }
    /**
     * Initialize SuperBit algorithm without parameters
     * (used only for serialization).
     */
    public SuperBit() {}
    /**
     * Compute the signature of this vector.
     * @param vector
     * @return
     */
    public final boolean[] signature(final double[] vector) {
        boolean[] sig = new boolean[this.hyperplanes.length];
        for (int i = 0; i < this.hyperplanes.length; i++) {
            sig[i] = (dotProduct(this.hyperplanes[i], vector) >= 0);
        }
        return sig;
    }
    /**
     * Compute the signature of the given FloatTensor, set the encapsulated signature for serialization
     * @param vector The target FloatTensor
     */
    public final void signature(final FloatTensor vector) {
        sig = new boolean[this.hyperplanes.length];
        for (int i = 0; i < this.hyperplanes.length; i++) {
            sig[i] = (dotProduct(this.hyperplanes[i], vector) >= 0);
        }
    }
    public final boolean[] getSignature() {
    	return sig;
    }
    /**
     * Compute the similarity between two signature, which is also an
     * estimation of the cosine similarity between the two vectors.
     * @param sig1
     * @param sig2
     * @return estimated cosine similarity
     */
    public final double similarity(final boolean[] sig1, final boolean[] sig2) {
        DoubleAdder agg = new DoubleAdder(); // Thread-safe accumulator
        Parallel.parallelFor(0, sig1.length, t -> {
            if (sig1[t] == sig2[t]) {
                agg.add(1); // Efficient atomic addition
            }
        });
        double sim = agg.sum() / sig1.length; // Use .sum() instead of .get()
        return Math.cos((1 - sim) * Math.PI);
    }

    /**
     * Get the hyperplanes coefficients used to compute signatures.
     * @return
     */
    public final double[][] getHyperplanes() {
        return this.hyperplanes;
    }
    /**
     * Computes the cosine similarity, computed as v1 dot v2 / (|v1| * |v2|).
     * Cosine similarity of two vectors is the cosine of the angle between them.
     * It ranges between -1 and +1
     *
     * @param v1
     * @param v2
     * @return
     */
    public static double cosineSimilarity(final double[]v1, final double[] v2) {
        return dotProduct(v1, v2) / (norm(v1) * norm(v2));
    }
    private static double[] product(final double x, final double[] v) {
        double[] r = new double[v.length];
        Parallel.parallelFor(0, v.length, t -> {
        	 r[t] = x * v[t];
        }); 
        return r;
    }
    private static double[] sub(final double[] a, final double[] b) {
        double[] r = new double[a.length];
        Parallel.parallelFor(0, a.length, t -> {
        	r[t] = a[t] - b[t];
        }); 
        return r;
    }
    private static void normalize(final double[] vector) {
        final double norm = norm(vector);
        Parallel.parallelFor(0, vector.length, t -> vector[t] /= norm);
    }
    /**
     * Returns the norm L2. sqrt(sum_i(v_i^2))
     * @param v
     * @return
     */
    private static double norm(final double[] v) {
        DoubleAdder agg = new DoubleAdder();
        Parallel.parallelFor(0, v.length, t -> agg.add(v[t] * v[t]));
        return Math.sqrt(agg.sum());
    }
    private static double dotProduct(final double[] v1, final double[] v2) {
        if (v1.length < 10_000) { // Adjust threshold based on benchmarking
            return IntStream.range(0, v1.length).mapToDouble(t -> v1[t] * v2[t]).sum();
        } else {
            DoubleAdder agg = new DoubleAdder();
            Parallel.parallelFor(0, v1.length, t -> agg.add(v1[t] * v2[t]));
            return agg.sum();
        }
    }
    private static double dotProduct(final double[] v1, final FloatTensor v2) {
        if (v1.length < 10_000) { // Adjust threshold based on benchmarking
            return IntStream.range(0, v1.length).mapToDouble(t -> v1[t] * v2.getFloat(t)).sum();
        } else {
            DoubleAdder agg = new DoubleAdder();
            Parallel.parallelFor(0, v1.length, t -> agg.add(v1[t] * v2.getFloat(t)));
            return agg.sum();
        }
    } 
	@Override
	public int compareTo(Object o) {
		return Double.compare(similarity(this.sig, ((SuperBit)o).sig), 1.0); // Ensures proper ordering
	}
}
