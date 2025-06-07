package com.neocoretechs.lsh.families;

import java.util.Comparator;

import com.neocoretechs.wordembedding.FloatTensor;

/**
 * This comparator can be used to sort candidate neighbors according to their
 * distance to a query vector. Either for linear search or to sort the LSH
 * candidates found in colliding hash bins.
 * 
 */
public class DistanceComparator implements Comparator<FloatTensor>{
	private final FloatTensor query;	
	/**
	 * @param query
	 */
	public DistanceComparator(FloatTensor query){
		this.query = query;
	}
	
	public double distance(FloatTensor one, FloatTensor other) {
		double distance=0;
		double similarity = one.dot(0,other,0,other.size()) / Math.sqrt(one.dot(0,one,0,one.size()) * other.dot(0,other,0,other.size()));
		distance = 1 - similarity;
		return distance;
	}
	
	/* (non-Javadoc)
	 * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
	 */
	@Override
	public int compare(FloatTensor one, FloatTensor other) {
		Double oneDistance = distance(query,one);
		Double otherDistance = distance(query,other);
		return oneDistance.compareTo(otherDistance);
	}
}
