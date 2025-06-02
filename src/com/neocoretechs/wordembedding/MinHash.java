package com.neocoretechs.wordembedding;
import java.util.HashSet;
import java.util.Set;
/**
 * MinHash is a hashing scheme that tents to produce similar signatures for sets
 * that have a high Jaccard similarity.
 *
 * The Jaccard similarity between two sets is the relative number of elements
 * these sets have in common: J(A, B) = |A ∩ B| / |A ∪ B| A MinHash signature is
 * a sequence of numbers produced by multiple hash functions hi. It can be shown
 * that the Jaccard similarity between two sets is also the probability that
 * this hash result is the same for the two sets: J(A, B) = Pr[hi(A) = hi(B)].
 * Therefore, MinHash signatures can be used to estimate Jaccard similarity
 * between two sets. Moreover, it can be shown that the expected estimation
 * error is O(1 / sqrt(n)), where n is the size of the signature (the number of
 * hash functions that are used to produce the signature).
 * Initialize the hash function for an similarity error of 0.1:
	 * For sets built from a dictionary of 5 items
	 * MinHash minhash = new MinHash(0.1, 5);
 */
public class MinHash implements java.io.Serializable, Comparable {
	private static final long serialVersionUID = -1L;
    private static final int LARGE_PRIME = 2147483647; // = 2^31 - 1 !
    private static final double THRESHOLD = 0.5;
    private int[] sig;
    /**
     * Compute the jaccard index between two sets.
     * @param s1
     * @param s2
     * @return
     */
    public static double jaccardIndex(final Set<Integer> s1, final Set<Integer> s2) {
        Set<Integer> intersection = new HashSet<Integer>(s1);
        intersection.retainAll(s2);
        Set<Integer> union = new HashSet<Integer>(s1);
        union.addAll(s2);
        if (union.isEmpty()) {
            return 0;
        }
        return (double) intersection.size() / union.size();
    }
    /**
     * Compute the exact jaccard index between two sets, represented as
     * arrays of booleans.
     * @param s1
     * @param s2
     * @return
     */
    public static double jaccardIndex(final boolean[] s1, final boolean[] s2) {
        if (s1.length != s2.length) {
            throw new InvalidParameterException("sets must be same size!");
        }
        return jaccardIndex(convert2Set(s1), convert2Set(s2));
    }
    /**
     * Convert a set represented as an array of booleans to a set of integer.
     *
     * @param array
     * @return
     */
    public static Set<Integer> convert2Set(final boolean[] array) {
        Set<Integer> set = new TreeSet<Integer>();
        for (int i = 0; i < array.length; i++) {
            if (array[i]) {
                set.add(i);
            }
        }
        return set;
    }
    public static Set<Integer> convert2Set(final FloatTensor array) {
        Set<Integer> set = new TreeSet<Integer>();
        for (int i = 0; i < array.size(); i++) {
                set.add(Float.floatToRawIntBits(array.getFloat(i)));
        }
        return set;
    }
    /**
     * Computes the size of the signature required to achieve a given error in
     * similarity estimation. (1 / error^2)
     *
     * @param error
     * @return size of the signature
     */
    public static int size(final double error) {
        if (error < 0 && error > 1) {
            throw new IllegalArgumentException("error should be in [0 .. 1]");
        }
        return (int) (1 / (error * error));
    }
    /**
     * Compute the size of the signature according to "Mining of Massive
     * Datasets" p88.
     * It can be shown that, using MinHash, the probability that the
     * signatures of 2 sets with Jaccard similarity s agree in all the
     * rows of at least one stage (band), and therefore become a candidate
     * pair, is 1−(1−s^R)^b
     * where R = signature_size / b (number of rows in a stage/band)
     * Thus, the curve that shows the probability that 2 items fall in the
     * same bucket for at least one of the stages, as a function of their
     * Jaccard index similarity, has a S shape.
     * The threshold (the value of similarity at which the probability of
     * becoming a candidate is 1/2) is a function of the number of stages
     * (s, or bands b in the book) and the signature size:
     * threshold ≃ (1/s)^(1/R)
     * Hence the signature size can be computed as:
     * R = ln(1/s) / ln(threshold)
     * signature_size = R * b
     */
    public static int computeSignatureSize(final int s) {
        int r = (int) Math.ceil(Math.log(1.0 / s) / Math.log(THRESHOLD)) + 1;
        return r * s;
    }
    /**
     * Signature size.
     */
    private int n;
    /**
     * Random a and b coefficients for the random hash functions.
     */
    private long[][] hash_coefs;
    /**
     * Dictionary size (is also the size of vectors if the sets are provided
     * as vectors).
     */
    private int dict_size;
    
    public MinHash(FloatTensor vector) {
    	// signature size, AKA similarity index, given as a value of .1 in example
    	init(size(0.1), vector.size(), new Random(8675309L));
    	sig = signature(convert2Set(vector));
    }
    /**
     * Initializes hash functions to compute MinHash signatures for sets built
     * from a dictionary of dict_size elements.
     *
     * @param size the number of hash functions (and the size of resulting
     * signatures)
     * @param dict_size
     */
    public MinHash(final int size, final int dict_size) {
        init(size, dict_size, new Random());
    }
    /**
     * Initializes hash function to compute MinHash signatures for sets built
     * from a dictionary of dict_size elements, with a given similarity
     * estimation error.
     *
     * @param error
     * @param dict_size
     */
    public MinHash(final double error, final int dict_size) {
        init(size(error), dict_size, new Random());
    }
    /**
     * Initializes hash functions to compute MinHash signatures for sets built
     * from a dictionary of dict_size elements.
     *
     * @param size the number of hash functions (and the size of resulting
     * signatures)
     * @param dict_size
     * @param seed random number generator seed. using the same value will
     * guarantee identical hashes across object instantiations
     */
    public MinHash(final int size, final int dict_size, final long seed) {
        init(size, dict_size, new Random(seed));
    }
    /**
     * Initializes hash function to compute MinHash signatures for sets built
     * from a dictionary of dict_size elements, with a given similarity
     * estimation error.
     *
     * @param error
     * @param dict_size
     * @param seed random number generator seed. using the same value will
     * guarantee identical hashes across object instantiations
     */
    public MinHash(final double error, final int dict_size, final long seed) {
        init(size(error), dict_size, new Random(seed));
    }
    /**
     * Computes the signature for this set The input set is represented as an
     * vector of booleans.
     * For example the array [true, false, true, true, false]
     * corresponds to the set {0, 2, 3}
     *
     * @param vector
     * @return the signature
     */
    public final int[] signature(final boolean[] vector) {
        if (vector.length != dict_size) {
            throw new IllegalArgumentException(
                    "Size of array should be dict_size");
        }
        return signature(convert2Set(vector));
    }
    
    public int[] getSignature() {
    	return sig;
    }
    /**
     * Computes the signature for this set. For example set = {0, 2, 3}
     *
     * @param set
     * @return the signature
     */
    public final int[] signature(final Set<Integer> set) {
        int[] sig = new int[n];
        for (int i = 0; i < n; i++) {
            sig[i] = Integer.MAX_VALUE;
        }
        // For each row r:
        //for (int r = 0; r < dict_size; r++) {
        // if set has 0 in row r, do nothing
        //    if (!set.contains(r)) {
        //        continue;
        //    }
        // Loop over true values, instead of loop over all values of dictionary
        // to speedup computation
        final List<Integer> list = new ArrayList<Integer>(set);
        Collections.sort(list);
        for (final int r : list) {
            // However, if c has 1 in row r, then for each i = 1, 2, . . . ,n
            // set SIG(i, c) to the smaller of the current value of
            // SIG(i, c) and hi(r)
            for (int i = 0; i < n; i++) {
                sig[i] = Math.min(sig[i], h(i, r));
            }
        }
        return sig;
    }
    /**
     * Computes an estimation of Jaccard similarity (the number of elements in
     * common) between two sets, using the MinHash signatures of these two sets.
     *
     * @param sig1 MinHash signature of set1
     * @param sig2 MinHash signature of set2 (produced using the same
     * coefficients)
     * @return the estimated similarity
     */
    public final double similarity(final int[] sig1, final int[] sig2) {
        if (sig1.length != sig2.length) {
            throw new IllegalArgumentException(
                    "Size of signatures should be the same");
        }
        double sim = 0;
        for (int i = 0; i < sig1.length; i++) {
            if (sig1[i] == sig2[i]) {
                sim += 1;
            }
        }
        return sim / sig1.length;
    }
    /**
     * Computes the expected error of similarity computed using signatures.
     *
     * @return the expected error
     */
    public final double error() {
        return 1.0 / Math.sqrt(n);
    }
    /**
     * Compute hash function coefficients using provided Random.
     * @param size
     * @param dict_size
     * @param r
     */
    private void init(final int size, final int dict_size, final Random r) {
        if (size <= 0) {
            throw new InvalidParameterException(
                    "Signature size should be positive");
        }
        if (dict_size <= 0) {
            throw new InvalidParameterException(
                    "Dictionary size (or vector size) should be positive");
        }
        // In function h(i, x) the largest value could be
        // dict_size * dict_size + dict_size
        // throw an error if dict_size * dict_size + dict_size > Long.MAX_VALUE
        if (dict_size > (Long.MAX_VALUE - dict_size) / dict_size) {
            throw new InvalidParameterException(
                    "Dictionary size (or vector size) is too big and will "
                            + "cause a multiplication overflow");
        }
        this.dict_size = dict_size;
        this.n = size;
        // h = (a * x) + b
        // a and b should be randomly generated in [1,PRIME-1]
        hash_coefs = new long[n][2];
        for (int i = 0; i < n; i++) {
            hash_coefs[i][0] = r.nextInt(LARGE_PRIME - 1) + 1; // a
            hash_coefs[i][1] = r.nextInt(LARGE_PRIME - 1) + 1; // b
        }
    }
    /**
     * Computes hi(x) as (a_i * x + b_i) % LARGE_PRIME .
     *
     * @param i
     * @param x
     * @return the hashed value of x, using ith hash function
     */
    private int h(final int i, final int x) {
        return (int)((hash_coefs[i][0] * (long) x + hash_coefs[i][1]) % LARGE_PRIME);
    }
    /**
     * Get the coefficients used by hash function hi.
     * @return
     */
    public final long[][] getCoefficients() {
        return hash_coefs;
    }
    
	@Override
	public int compareTo(Object o) {
		return Double.compare(similarity(this.sig, ((MinHash)o).sig), 1.0); // Ensures proper ordering
	}
}

