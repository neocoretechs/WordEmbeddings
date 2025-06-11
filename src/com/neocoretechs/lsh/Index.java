package com.neocoretechs.lsh;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import com.neocoretechs.lsh.families.CosineHash;
import com.neocoretechs.lsh.families.DistanceComparator2;
import com.neocoretechs.wordembedding.FileUtils;
import com.neocoretechs.wordembedding.FloatTensor;

/**
 * The index makes it easy to store vectors and lookup queries efficiently. For
 * the moment the index is stored in memory. It holds a number of hash tables,
 * each with a couple of hashes. Together they can be used for efficient lookup
 * of nearest neighbors.
 * {@link HashTable}
 * 
 */
public class Index implements Serializable{
	private static final long serialVersionUID = 3757702142917691272L;
	private static boolean DEBUG = true;
	private final static Logger LOG = Logger.getLogger(Index.class.getName()); 
	private static final int VECTOR_DIMENSION = 50;
	//GLOVE_FILE = "glove.6B.50d.txt";
	public static int numberOfHashTables = 8;
	public static int numberOfHashes = 8;
	public static int numberOfNeighbors = -1;//4;

	private CosineHash family;
	private List<HashTable> hashTable; 
	private int evaluated;
	
	public Index() {}
	/**
	 * Create a new index.
	 * 
	 * @param family
	 *            The family of hash functions to use.
	 * @param numberOfHashes
	 *            The number of hashes that are concatenated in each hash table.
	 *            More concatenated hashes means that less candidates are
	 *            selected for evaluation.
	 * @param numberOfHashTables
	 *            The number of hash tables in use, recall increases with the
	 *            number of hash tables. Memory use also increases. Time needed
	 *            to compute a hash also increases marginally.
	 */
	public Index(int numberOfHashes, int numberOfHashTables, int projectionVectorSize){
		hashTable = new ArrayList<HashTable>();
		for(int i = 0 ; i < numberOfHashTables ; i++ ){
			hashTable.add(new HashTable(i, numberOfHashes, projectionVectorSize));
		}
		evaluated = 0;
	}
	
	/**
	 * Add a vector to the current index. The hashes are calculated with the
	 * current hash family and added in the right place.
	 * 
	 * @param vector
	 *            The vector to add.
	 */
	public void index(FloatTensor vector) {
		for (HashTable table : hashTable) {
			table.add(vector);
		}
	}
	
	/**
	 * The number of hash tables used in the current index.
	 * @return The number of hash tables used in the current index.
	 */
	public int getNumberOfHashTables(){
		return hashTable.size();
	}
	
	/**
	 * The number of hashes used in each hash table in the current index.
	 * @return The number of hashes used in each hash table in the current index.
	 */
	public int getNumberOfHashes(){
		return hashTable.get(0).getNumberOfHashes();
	}

	/**
	 * Query for the k nearest neighbors in using the current index. The
	 * performance (in computing time and recall/precision) depends mainly on
	 * how the current index is constructed and how the underlying data looks.
	 * 
	 * @param query
	 *            The query vector. The center of the neighborhood.
	 * @param maxSize
	 *            The maximum number of neighbors to return or -1. Beware, the number
	 *            of neighbors returned lays between zero and the chosen
	 *            maximum.
	 * @return A list of nearest neighbors, the number of neighbors returned
	 *         lays between zero and a chosen maximum.
	 */
	public List<FloatTensor> query(final FloatTensor query, int maxSize){
		Set<FloatTensor> candidateSet = new HashSet<FloatTensor>();
		for(HashTable table : hashTable) {
			if(DEBUG)
				LOG.info(table.toString());
			List<FloatTensor> v = table.query(query);
			if(DEBUG)
				LOG.info("returned "+v.size()+" elements");
			candidateSet.addAll(v);
		}
		List<FloatTensor> candidates = new ArrayList<FloatTensor>(candidateSet);
		evaluated += candidates.size();
		if(DEBUG)
			LOG.info("evaluated:"+evaluated);
		DistanceComparator2 dc = new DistanceComparator2(query);
		Collections.sort(candidates,dc);
		if(maxSize > 0 && candidates.size() > maxSize){
			candidates = candidates.subList(0, maxSize);
		}
		return candidates;
	}
	
	/**
	 * The number of near neighbor candidates that are evaluated during the queries on this index. 
	 * Can be used to calculate the average evaluations per query.
	 * @return The number of near neighbor candidates that are evaluated during the queries on this index. 
	 */
	public int getTouched(){
		return evaluated;
	}
	
	/**
	 * Serializes the index to disk.
	 * @param index the storage object.
	 */
	public static void serialize(Index index){
		try {
			String serializationFile = serializationName(index.getNumberOfHashes(), index.getNumberOfHashTables());
			OutputStream file = new FileOutputStream(serializationFile);
			OutputStream buffer = new BufferedOutputStream(file);
			ObjectOutput output = new ObjectOutputStream(buffer);
			try {
				output.writeObject(index);
			} finally {
				output.close();
			}
		} catch (IOException ex) {

		}
	}
	
	/**
	 * Return a unique name for a hash table wit a family and number of hashes. 
	 * @param hashtable the hash table.
	 * @return e.g. "com.neocoretechs.lsh.families.CosineHashFamily_4_4.bin"
	 */
	private static String serializationName(int numberOfHashes,int numberOfHashTables){
		return "CosineHash_" + numberOfHashes + "_" + numberOfHashTables + ".bin";
	}
	
	/**
	 * Deserializes the hash table from disk. If deserialization fails, 
	 * a new Index is created.
	 * 
	 * @param family The family.
	 * @param numberOfHashes the number of hashes.
	 * @param numberOfHashTables The number of hash tables
	 * @return a new, or deserialized object.
	 */
	public static Index deserialize(int numberOfHashes,int numberOfHashTables){
		Index index = null;
		String serializationFile = serializationName(numberOfHashes, numberOfHashTables);
		if(FileUtils.exists(serializationFile)){
			try {		
				InputStream file = new FileInputStream(serializationFile);
				InputStream buffer = new BufferedInputStream(file);
				ObjectInput input = new ObjectInputStream(buffer);
				try {
					index = (Index) input.readObject();
				} finally {
					input.close();
				}
			} catch (ClassNotFoundException ex) {
				LOG.severe("Could not find class during deserialization: " + ex.getMessage());
			} catch (IOException ex) {
				LOG.severe("IO exeption during during deserialization: " + ex.getMessage());
				ex.printStackTrace();
			}
		}
		if(index == null)
			index = new Index(numberOfHashes,numberOfHashTables,VECTOR_DIMENSION);
		return index;
	}

}


