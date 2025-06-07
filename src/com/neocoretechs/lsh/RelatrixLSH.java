package com.neocoretechs.lsh;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import com.neocoretechs.lsh.families.CosineHash;
import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.Relatrix;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.wordembedding.FloatTensor;
import com.neocoretechs.wordembedding.Parallel;

/**
 * An {@link Index} contains one or more locality sensitive hash tables. These hash
 * tables contain the mapping between a combination of a number of hashes
 * (encoded using an integer) and a list of possible nearest neighbors.<p>
 *
 * A hash function can hash a vector of arbitrary dimensions to an integer
 * representation. The hash function needs to be locality sensitive to work in
 * the locality sensitive hash scheme. Meaning that vectors that are 'close'
 * according to some metric have a high probability to end up with the same
 * hash.<p>
 * In the context of Locality-Sensitive Hashing (LSH), w represents the bucket width or window size.<p>
 * When we compute the hash value for a vector using a random projection. Here's what each component does:<p>
 * vector.dot(randomProjection): Computes the dot product of the input vector and a random projection vector. <br>
 * This projects the input vector onto a random direction.<br>
 * offset: Adds a random offset to the projected value. <br>
 * This helps to shift the projected values and create a more uniform distribution.<p>
 * w: The bucket width or window size. This value determines the granularity of the hash function.<br>
 * By dividing the projected value (plus offset) by w, you're essentially:<br>
 * Quantizing the projected values into discrete buckets.<br>
 * Assigning each bucket a unique hash value. <br>
 * The choice of w affects the trade-off between:<br>
 * Precision: Smaller w values result in more precise hashing, but may lead to more collisions.
 * Larger w values result in fewer collisions, but may reduce precision.<p>
 * In general, w is a hyperparameter that needs to be tuned for specific applications and datasets. 
 * A good choice of w can significantly impact the performance of the LSH algorithm.<p>
 * This class is designed to be stored in the Relatrix database to serve as a template for encoding and retrieving
 * a given set of floating point tensors.
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 */
public class RelatrixLSH implements Serializable {
	private static final long serialVersionUID = -5410017645908038641L;
	private static boolean DEBUG = true;
	public static final int VECTOR_DIMENSION = 50;
	public static int numberOfHashTables = 8;
	public static int numberOfHashes = 8;
	private static int radius = 500;
	/**
	 * Contains the mapping between a combination of a number of hashes (encoded
	 * using an integer) and a list of possible nearest neighbours
	 */
	private CosineHash[] hashFunctions;
	private CosineHash family;
	private int index;
	
	public RelatrixLSH() {}
	/**
	 * Initialize a new hash table, it needs a hash family and a number of hash
	 * functions that should be used.
	 * 
	 * @param numberOfHashes
	 *            The number of hash functions that should be used.
	 * @param family
	 *            The hash function family knows how to create new hash
	 *            functions, and is used therefore.
	 */
	public RelatrixLSH(int index, int numberOfHashes, int projectionVectorSize) {
		this.index = index;
	    this.hashFunctions = new CosineHash[numberOfHashes];
	    if(numberOfHashes > 64)
	    	Parallel.parallelFor(0, numberOfHashes, i -> {
	    		hashFunctions[i] = new CosineHash(projectionVectorSize);
	    	});
	    else
	    	for(int i = 0; i < numberOfHashes; i++)
	    		hashFunctions[i] = new CosineHash(projectionVectorSize);
	}

	/**
	 * Query the hash table for a vector. It calculates the hash for the vector,
	 * and does a lookup in the hash table. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query
	 *            The query vector.
	 * @return Does a lookup in the table for a query using its hash. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned.
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public List<FloatTensor> query(FloatTensor query) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException {
		Integer combinedHash = hash(query);
		if(DEBUG)
			System.out.println("Combined hash for query:"+combinedHash);
		Iterator<?> it = Relatrix.findSet(combinedHash, index, '?');
		ArrayList<FloatTensor> res = new ArrayList<FloatTensor>();
		while(it.hasNext()) {
			Result r= (Result) it.next();
			res = (ArrayList<FloatTensor>)r.get();
		}
		return res;
	}

	/**
	 * Add a vector to the index.
	 * @param vector
	 * @throws DuplicateKeyException 
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 */
	public void add(FloatTensor vector) throws IllegalAccessException, ClassNotFoundException, IOException, DuplicateKeyException {
		Integer combinedHash = hash(vector);
		Relatrix.store(combinedHash, index, vector);
	}
	
	/**
	 * Calculate the combined hash for a vector.
	 * @param vector The vector to calculate the combined hash for.
	 * @return An integer representing a combined hash.
	 */
	private Integer hash(FloatTensor vector){
		int hashes[] = new int[hashFunctions.length];
		for(int i = 0 ; i < hashFunctions.length ; i++){
			hashes[i] = hashFunctions[i].hash(vector);
		}
		Integer combinedHash = family.combine(hashes);
		return combinedHash;
	}

	/**
	 * Return the number of hash functions used in the hash table.
	 * @return The number of hash functions used in the hash table.
	 */
	public int getNumberOfHashes() {
		return hashFunctions.length;
	}

	@Override
	public String toString() {
		return String.format("%s index=%d family=%s hashes=%s",this.getClass().getName(), index, family, Arrays.toString(hashFunctions));
	}
}
